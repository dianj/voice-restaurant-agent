"""Tools the agent can call. Every tool validates its input and returns a JSON string.

Result status values: "success", "unavailable" (valid request, but no table: alternatives are included)
and "error" (invalid input; the message says what to fix).
"""
import json
import random
import sqlite3
from contextlib import contextmanager
from datetime import date as date_cls, datetime, timedelta

from . import config
from .data import ALLERGENS, CUISINES, LOCATIONS

# Words clients use for the 14 declared allergens.
ALLERGEN_SYNONYMS = {
    "dairy": "milk", "lactose": "milk", "cheese": "milk", "wheat": "gluten", "egg": "eggs",
    "peanut": "peanuts", "tree nuts": "nuts", "nut": "nuts", "walnuts": "nuts", "shellfish": "crustaceans",
    "prawns": "crustaceans", "shrimp": "crustaceans", "sulfites": "sulphites", "sesame seeds": "sesame",
    "soya": "soy", "mussels": "molluscs", "octopus": "molluscs", "squid": "molluscs",
}
# Letters and digits that cannot be confused when read aloud or typed (no O/0, I/1).
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def now() -> datetime:
    """Current local time. A function so tests can freeze it."""
    return datetime.now()


def connect():
    conn = sqlite3.connect(config.DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db():
    """Connection that commits on success, rolls back on error and is always closed."""
    conn = connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def result(status: str, **fields) -> str:
    return json.dumps({"status": status, **fields}, ensure_ascii=False)


def error(message: str) -> str:
    return result("error", message=message)


# --- Helpers ----------------------------------------------------------------------------

def minutes(hhmm: str) -> int:
    hours, mins = hhmm.split(":")
    return int(hours) * 60 + int(mins)


def hhmm(total_minutes: int) -> str:
    total_minutes %= 24 * 60
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def opening_window(opening_hours: str):
    """'12:00-00:00' -> (720, 1440): closing at or before opening means after midnight."""
    opens, closes = (minutes(part.strip()) for part in opening_hours.split("-"))
    if closes <= opens:
        closes += 24 * 60
    return opens, closes


def match_choice(value: str, choices) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    return next((c for c in choices if c.lower() == lowered), None)


def get_restaurant(conn, restaurant_id) -> sqlite3.Row | None:
    if not isinstance(restaurant_id, str):
        return None
    return conn.execute("SELECT * FROM restaurants WHERE restaurant_id = ?",
                        (restaurant_id.strip().upper(),)).fetchone()


def describe_rating(rating) -> float | str:
    return rating if rating is not None else "not rated yet"


def parse_booking_request(restaurant, date_text, time_text, guests):
    """Validate date, time and party size against the rules and the opening hours.

    Returns (booking_datetime, minute_of_day, None) or (None, None, error_message).
    """
    if isinstance(guests, bool) or not isinstance(guests, int):
        return None, None, "guests must be a whole number."
    if guests < config.MIN_GUESTS:
        return None, None, "guests must be at least 1."
    if guests > config.MAX_GUESTS:
        return None, None, (f"Online bookings are limited to {config.MAX_GUESTS} guests. "
                            "Larger groups are handled by the events team: offer to pass the request to a human.")
    try:
        day = date_cls.fromisoformat(str(date_text).strip())
    except ValueError:
        return None, None, "date must be in YYYY-MM-DD format."
    try:
        start = minutes(str(time_text).strip())
        if not (0 <= start < 24 * 60):
            raise ValueError
    except ValueError:
        return None, None, "time must be in HH:MM format (24-hour clock)."
    if start % config.SLOT_STEP_MINUTES:
        return None, None, f"Bookings start on the hour or half hour (e.g. {hhmm(start - start % 30)} or {hhmm(start - start % 30 + 30)})."

    opens, closes = opening_window(restaurant["opening_hours"])
    last_seating = closes - config.LAST_SEATING_BEFORE_CLOSE
    # A time after midnight belongs to the previous evening when the restaurant is open past midnight.
    if start < opens and start + 24 * 60 <= last_seating:
        start += 24 * 60
    if not (opens <= start <= last_seating):
        return None, None, (f"{restaurant['name']} is open {restaurant['opening_hours']}; "
                            f"bookings are possible from {hhmm(opens)} to {hhmm(last_seating)}.")

    when = datetime.combine(day, datetime.min.time()) + timedelta(minutes=start)
    current = now()
    if when <= current:
        return None, None, f"That time is in the past (now it is {current:%Y-%m-%d %H:%M})."
    if (day - current.date()).days > config.MAX_DAYS_AHEAD:
        return None, None, f"Bookings can be made at most {config.MAX_DAYS_AHEAD} days ahead."
    return when, start, None


def seats_taken(conn, restaurant_id: str, day: str, start: int) -> int:
    """Guests already seated at `start` (minute of the booking day) by overlapping confirmed reservations."""
    rows = conn.execute(
        "SELECT date, time, guests FROM reservations "
        "WHERE restaurant_id = ? AND status = 'confirmed' AND date IN (?, ?, ?)",
        (restaurant_id, day,
         (date_cls.fromisoformat(day) - timedelta(days=1)).isoformat(),
         (date_cls.fromisoformat(day) + timedelta(days=1)).isoformat()),
    ).fetchall()
    taken = 0
    base = date_cls.fromisoformat(day)
    for row in rows:
        other = (date_cls.fromisoformat(row["date"]) - base).days * 24 * 60 + minutes(row["time"])
        if abs(other - start) < config.SITTING_MINUTES:
            taken += row["guests"]
    return taken


def booking_minute(day: str, when: datetime) -> int:
    """Minute offset of `when` from midnight at the start of `day` (can exceed 1440 after midnight)."""
    return int((when - datetime.combine(date_cls.fromisoformat(day), datetime.min.time())).total_seconds() // 60)


def alternative_times(conn, restaurant, day: str, wanted: int, guests: int, limit: int = 3) -> list:
    """Closest free start times on the same day."""
    opens, closes = opening_window(restaurant["opening_hours"])
    current = now()
    base = datetime.combine(date_cls.fromisoformat(day), datetime.min.time())
    candidates = range(opens, closes - config.LAST_SEATING_BEFORE_CLOSE + 1, config.SLOT_STEP_MINUTES)
    free = []
    for start in sorted(candidates, key=lambda m: (abs(m - wanted), m)):
        if start == wanted or base + timedelta(minutes=start) <= current:
            continue
        if seats_taken(conn, restaurant["restaurant_id"], day, start) + guests <= restaurant["seating_capacity"]:
            free.append(hhmm(start))
        if len(free) == limit:
            break
    return free


def new_code(conn) -> str:
    rng = random.SystemRandom()
    while True:
        code = "R-" + "".join(rng.choice(CODE_ALPHABET) for _ in range(5))
        if not conn.execute("SELECT 1 FROM reservations WHERE code = ?", (code,)).fetchone():
            return code


def normalize_code(code) -> str:
    text = str(code or "").strip().upper().replace(" ", "")
    if text and not text.startswith("R-"):
        text = "R-" + text.removeprefix("R")
    return text


# --- Tools ------------------------------------------------------------------------------

def search_restaurants(cuisine: str = None, city: str = None, area: str = None, min_rating: float = None,
                       open_at: str = None, limit: int = 5):
    clauses, params = [], []
    if cuisine:
        match = match_choice(cuisine, CUISINES)
        if not match:
            return error(f"Unknown cuisine '{cuisine}'. Available: {CUISINES}.")
        clauses.append("cuisine = ?")
        params.append(match)
    if city:
        match = match_choice(city, LOCATIONS)
        if not match:
            return error(f"We have restaurants only in {list(LOCATIONS)}.")
        clauses.append("city = ?")
        params.append(match)
        city = match
    if area:
        areas = LOCATIONS[city] if city else sorted({a for areas in LOCATIONS.values() for a in areas})
        match = match_choice(area, areas)
        if not match:
            return error(f"Unknown area '{area}'. Areas{' in ' + city if city else ''}: {areas}.")
        clauses.append("area = ?")
        params.append(match)
    if min_rating is not None:
        if isinstance(min_rating, bool) or not isinstance(min_rating, (int, float)) or not 0 <= min_rating <= 5:
            return error("min_rating must be a number between 0 and 5.")
        clauses.append("rating >= ?")
        params.append(min_rating)
    if open_at is not None:
        try:
            wanted = minutes(str(open_at).strip())
        except ValueError:
            return error("open_at must be in HH:MM format.")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10:
        limit = 5

    where = " AND ".join(clauses) or "1 = 1"
    with db() as conn:
        rows = conn.execute(
            f"SELECT restaurant_id, name, city, area, cuisine, price_range, rating, opening_hours "
            f"FROM restaurants WHERE {where} ORDER BY rating IS NULL, rating DESC, name", params
        ).fetchall()
    if open_at is not None:
        def is_open(row):
            opens, closes = opening_window(row["opening_hours"])
            return opens <= wanted < closes or opens <= wanted + 24 * 60 < closes
        rows = [r for r in rows if is_open(r)]
    if not rows:
        return result("success", total_matches=0, restaurants=[],
                      hint="No match. Suggest relaxing a filter (another area, cuisine or time).")
    top = [{**dict(r), "rating": describe_rating(r["rating"])} for r in rows[:limit]]
    return result("success", total_matches=len(rows), restaurants=top)


def get_menu(restaurant_id: str, avoid_allergens: list = None, vegetarian_only: bool = False, vegan_only: bool = False):
    avoid = []
    for item in avoid_allergens or []:
        name = ALLERGEN_SYNONYMS.get(str(item).strip().lower(), str(item).strip().lower())
        if name not in ALLERGENS:
            return error(f"Unknown allergen '{item}'. Declared allergens: {list(ALLERGENS)}.")
        avoid.append(name)
    with db() as conn:
        restaurant = get_restaurant(conn, restaurant_id)
        if not restaurant:
            return error(f"Restaurant '{restaurant_id}' not found. Use search_restaurants to find the ID.")
        rows = conn.execute(
            "SELECT dish_name, ingredients, allergens, vegetarian, vegan, price FROM menu_items "
            "WHERE restaurant_id = ? ORDER BY id", (restaurant["restaurant_id"],)
        ).fetchall()
    dishes = []
    for r in rows:
        allergens = [a.strip() for a in r["allergens"].split(",") if a.strip()]
        if vegetarian_only and not r["vegetarian"] or vegan_only and not r["vegan"]:
            continue
        dish = {"dish": r["dish_name"], "ingredients": r["ingredients"], "allergens": allergens or ["none declared"],
                "vegetarian": bool(r["vegetarian"]), "vegan": bool(r["vegan"]), "price_eur": r["price"]}
        if avoid:
            dish["contains_avoided"] = sorted(set(allergens) & set(avoid))
        dishes.append(dish)
    fields = {"restaurant": restaurant["name"], "restaurant_id": restaurant["restaurant_id"], "dishes": dishes}
    if avoid:
        fields["safe_dishes"] = [d["dish"] for d in dishes if not d["contains_avoided"]]
        fields["safety_note"] = ("Allergen data comes from the menu. Kitchens handle all allergens, so trace amounts "
                                 "are possible: the guest should tell the staff about the allergy when ordering.")
    return result("success", **fields)


def check_availability(restaurant_id: str, date: str, time: str, guests: int):
    with db() as conn:
        restaurant = get_restaurant(conn, restaurant_id)
        if not restaurant:
            return error(f"Restaurant '{restaurant_id}' not found. Use search_restaurants to find the ID.")
        when, start, problem = parse_booking_request(restaurant, date, time, guests)
        if problem:
            return error(problem)
        day = str(date).strip()
        free = restaurant["seating_capacity"] - seats_taken(conn, restaurant["restaurant_id"], day, booking_minute(day, when))
        base = {"restaurant": restaurant["name"], "restaurant_id": restaurant["restaurant_id"],
                "date": when.date().isoformat(), "time": when.strftime("%H:%M"), "guests": guests}
        if guests <= free:
            return result("success", available=True, **base)
        return result("unavailable", available=False, **base,
                      alternatives=alternative_times(conn, restaurant, day, booking_minute(day, when), guests))


def book_table(restaurant_id: str, customer_name: str, date: str, time: str, guests: int,
               phone: str = None, notes: str = None):
    if not isinstance(customer_name, str) or not 2 <= len(customer_name.strip()) <= 80:
        return error("customer_name is required (2-80 characters). Ask the guest for their name.")
    if phone is not None and (not isinstance(phone, str) or len(phone) > 30):
        return error("phone must be text of at most 30 characters.")
    if notes is not None and (not isinstance(notes, str) or len(notes) > 300):
        return error("notes must be text of at most 300 characters.")
    conn = connect()
    try:
        conn.isolation_level = None
        # Write lock before the capacity check, so two parallel bookings cannot both take the last seats.
        conn.execute("BEGIN IMMEDIATE")
        restaurant = get_restaurant(conn, restaurant_id)
        if not restaurant:
            conn.execute("ROLLBACK")
            return error(f"Restaurant '{restaurant_id}' not found. Use search_restaurants to find the ID.")
        when, start, problem = parse_booking_request(restaurant, date, time, guests)
        if problem:
            conn.execute("ROLLBACK")
            return error(problem)
        day = str(date).strip()
        rid = restaurant["restaurant_id"]
        minute = booking_minute(day, when)
        if seats_taken(conn, rid, day, minute) + guests > restaurant["seating_capacity"]:
            alternatives = alternative_times(conn, restaurant, day, minute, guests)
            conn.execute("ROLLBACK")
            return result("unavailable", available=False, restaurant=restaurant["name"],
                          date=when.date().isoformat(), time=when.strftime("%H:%M"), guests=guests,
                          alternatives=alternatives)
        code = new_code(conn)
        stamp = now().isoformat(timespec="seconds")
        conn.execute(
            "INSERT INTO reservations (code, restaurant_id, customer_name, phone, date, time, guests, notes, status, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'confirmed', ?, ?)",
            (code, rid, customer_name.strip(), (phone or "").strip() or None, when.date().isoformat(),
             when.strftime("%H:%M"), guests, (notes or "").strip() or None, stamp, stamp),
        )
        conn.execute("COMMIT")
    finally:
        conn.close()
    return result("success", reservation_code=code, restaurant=restaurant["name"],
                  address=f"{restaurant['area']}, {restaurant['city']}", date=when.date().isoformat(),
                  time=when.strftime("%H:%M"), guests=guests, customer_name=customer_name.strip())


def find_reservation(code: str):
    with db() as conn:
        row = conn.execute(
            "SELECT r.code, r.customer_name, r.date, r.time, r.guests, r.status, r.notes, s.name AS restaurant "
            "FROM reservations r JOIN restaurants s USING (restaurant_id) WHERE r.code = ?", (normalize_code(code),)
        ).fetchone()
    if not row:
        return error(f"No reservation with code '{code}'. Ask the guest to repeat the code (format R-XXXXX).")
    return result("success", reservation=dict(row))


def cancel_reservation(code: str, customer_name: str):
    code = normalize_code(code)
    with db() as conn:
        row = conn.execute("SELECT customer_name, status FROM reservations WHERE code = ?", (code,)).fetchone()
        if not row:
            return error(f"No reservation with code '{code}'. Ask the guest to repeat the code (format R-XXXXX).")
        # The name works as a simple check that the caller owns the booking.
        if str(customer_name or "").strip().lower() != row["customer_name"].lower():
            return error("The name does not match this reservation. Ask the guest for the name used when booking.")
        if row["status"] == "cancelled":
            return result("success", reservation_code=code, already_cancelled=True)
        conn.execute("UPDATE reservations SET status = 'cancelled', updated_at = ? WHERE code = ?",
                     (now().isoformat(timespec="seconds"), code))
    return result("success", reservation_code=code, cancelled=True)


def list_reservations(limit: int = 50) -> list:
    """Staff view: latest reservations (not a tool for the agent)."""
    with db() as conn:
        rows = conn.execute(
            "SELECT r.code, r.customer_name, r.phone, r.date, r.time, r.guests, r.notes, r.status, r.created_at, "
            "s.name AS restaurant, s.city, s.area FROM reservations r JOIN restaurants s USING (restaurant_id) "
            "ORDER BY r.id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
