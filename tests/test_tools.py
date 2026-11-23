"""Unit tests for the booking rules, search and menu tools (no LLM). The clock is frozen at Fri 2026-11-20 10:00."""
import json
import re

from restaurant import config, data, tools


def call(fn, *args, **kwargs):
    return json.loads(fn(*args, **kwargs))


def restaurant(where: str = "1 = 1", params=()):
    with tools.db() as conn:
        query = f"SELECT * FROM restaurants WHERE {where} ORDER BY restaurant_id LIMIT 1"
        return dict(conn.execute(query, params).fetchone())


TOMORROW = "2026-11-21"


# --- Demo data ---------------------------------------------------------------------------

def test_demo_data_is_deterministic_and_every_restaurant_has_a_menu(tmp_path):
    first, second = data.build_database(str(tmp_path / "a.db")), data.build_database(str(tmp_path / "b.db"))
    assert first == second and first["restaurants"] == 1000
    with tools.db() as conn:
        without_menu = conn.execute(
            "SELECT COUNT(*) FROM restaurants WHERE restaurant_id NOT IN (SELECT restaurant_id FROM menu_items)"
        ).fetchone()[0]
        names = conn.execute("SELECT COUNT(DISTINCT name) FROM restaurants").fetchone()[0]
    assert without_menu == 0
    assert names == 1000  # readable, unique names (no "#437" suffixes)


# --- Search ------------------------------------------------------------------------------

def test_search_filters_and_sorts_by_rating_with_unrated_last():
    result = call(tools.search_restaurants, cuisine="italian", city="sofia", limit=10)
    assert result["status"] == "success"
    rows = result["restaurants"]
    assert rows and all(r["cuisine"] == "Italian" and r["city"] == "Sofia" for r in rows)
    rated = [r["rating"] for r in rows if r["rating"] != "not rated yet"]
    assert rated == sorted(rated, reverse=True)


def test_search_rejects_unknown_values_with_the_allowed_list():
    assert "Available" in call(tools.search_restaurants, cuisine="Klingon")["message"]
    assert "Sofia" in call(tools.search_restaurants, city="Paris")["message"]


def test_search_open_at_handles_restaurants_open_past_midnight():
    # At 23:30 only the places open until midnight qualify ("11:30-23:30" has just closed).
    result = call(tools.search_restaurants, city="Sofia", open_at="23:30", limit=10)
    assert result["restaurants"]
    assert {r["opening_hours"] for r in result["restaurants"]} == {"12:00-00:00"}


# --- Menu and allergens ------------------------------------------------------------------

def test_menu_marks_dishes_with_avoided_allergens_and_accepts_everyday_words():
    rid = restaurant("cuisine = 'Italian'")["restaurant_id"]
    result = call(tools.get_menu, rid, avoid_allergens=["dairy", "wheat"])
    assert result["status"] == "success"
    for dish in result["dishes"]:
        assert set(dish["contains_avoided"]) == set(dish["allergens"]) & {"milk", "gluten"}
    assert result["safe_dishes"] == [d["dish"] for d in result["dishes"] if not d["contains_avoided"]]
    assert "staff" in result["safety_note"]


def test_menu_rejects_unknown_allergen_and_unknown_restaurant():
    assert call(tools.get_menu, "REST-0001", avoid_allergens=["kryptonite"])["status"] == "error"
    assert call(tools.get_menu, "REST-9999")["status"] == "error"


# --- Booking rules -----------------------------------------------------------------------

def test_booking_request_validation():
    r = restaurant("opening_hours = '12:00-00:00'")
    rid = r["restaurant_id"]
    cases = {
        ("2026-11-19", "19:00", 2): "past",            # yesterday
        ("2026-11-20", "09:00", 2): "open",            # before opening
        (TOMORROW, "23:30", 2): "open",                # after the last seating (close 00:00 - 60 min)
        (TOMORROW, "19:15", 2): "half hour",
        ("2027-03-01", "19:00", 2): "days ahead",
        (TOMORROW, "19:00", 0): "at least 1",
        (TOMORROW, "19:00", 25): "restaurant staff",
        (TOMORROW, "19:00", True): "whole number",
        ("21.11.2026", "19:00", 2): "YYYY-MM-DD",
        (TOMORROW, "7pm", 2): "HH:MM",
    }
    for (day, time, guests), expected in cases.items():
        result = call(tools.check_availability, rid, day, time, guests)
        assert result["status"] == "error", (day, time, guests)
        assert expected in result["message"], result["message"]
    assert call(tools.check_availability, rid, TOMORROW, "23:00", 2)["available"] is True  # last seating


def test_full_slot_is_unavailable_and_alternatives_are_free():
    r = restaurant("seating_capacity = 20 AND opening_hours = '10:00-22:00'")
    rid = r["restaurant_id"]
    assert call(tools.book_table, rid, "Maria", TOMORROW, "19:00", 18)["status"] == "success"
    blocked = call(tools.check_availability, rid, TOMORROW, "19:30", 4)  # overlaps the 19:00 sitting
    assert blocked["status"] == "unavailable" and blocked["alternatives"]
    for alt in blocked["alternatives"]:
        assert call(tools.check_availability, rid, TOMORROW, alt, 4)["available"] is True
    # A sitting lasts SITTING_MINUTES, so 21:00 is free again.
    assert config.SITTING_MINUTES == 120
    assert call(tools.check_availability, rid, TOMORROW, "21:00", 20)["available"] is True


def test_booking_is_saved_and_can_be_found_and_cancelled_by_its_owner():
    rid = restaurant()["restaurant_id"]
    hours_open = tools.opening_window(restaurant()["opening_hours"])[0]
    time = tools.hhmm(hours_open + 60)
    booking = call(tools.book_table, rid, "Alex Ivanov", TOMORROW, time, 2, notes="gluten allergy")
    assert booking["status"] == "success" and re.fullmatch(r"R[A-Z0-9]{5}", booking["reservation_code"])
    code = booking["reservation_code"]

    for variant in (code.lower(), f"R-{code[1:]}", " ".join(code), code[1:]):  # typed and spoken variants
        assert call(tools.find_reservation, variant)["reservation"]["code"] == code, variant
    found = call(tools.find_reservation, code)
    assert "Alex" not in json.dumps(found) and "customer_name" not in found["reservation"]
    # so a cancelled booking can be checked and booked again; the code alone does not reveal who booked
    assert found["reservation"]["restaurant_id"] == rid
    assert found["reservation"]["notes"] == "gluten allergy"

    assert call(tools.cancel_reservation, code, "Someone Else")["status"] == "error"
    assert call(tools.cancel_reservation, code, "alex ivanov")["cancelled"] is True
    assert call(tools.find_reservation, code)["reservation"]["status"] == "cancelled"


def test_owner_name_tolerates_speech_recognition_variants_but_not_another_name():
    row = {"customer_name": "Хано Споруха"}
    for heard in ("хано спаруха", "Хано с Паруха", "Hano Sporuha", "  ХАНО СПОРУХА. "):
        assert tools.is_owner(row, heard), heard
    for other in ("Мария Петрова", "Хано", "", "Иван Споруха"):
        assert not tools.is_owner(row, other), other
    assert tools.is_owner({"customer_name": "Ana"}, "ana") and not tools.is_owner({"customer_name": "Ana"}, "Ina")


def test_cancelled_reservation_frees_the_seats():
    r = restaurant("seating_capacity = 20 AND opening_hours = '10:00-22:00'")
    rid = r["restaurant_id"]
    code = call(tools.book_table, rid, "Group", TOMORROW, "19:00", 20)["reservation_code"]
    assert call(tools.check_availability, rid, TOMORROW, "19:00", 1)["available"] is False
    call(tools.cancel_reservation, code, "Group")
    assert call(tools.check_availability, rid, TOMORROW, "19:00", 20)["available"] is True


def test_booking_requires_a_name():
    assert "name" in call(tools.book_table, "REST-0001", " ", TOMORROW, "19:00", 2)["message"]


# --- Changing a reservation --------------------------------------------------------------

def book_at_10_to_22(name="Alex", time="19:00", guests=2):
    r = restaurant("seating_capacity = 20 AND opening_hours = '10:00-22:00'")
    booked = call(tools.book_table, r["restaurant_id"], name, TOMORROW, time, guests)
    return r["restaurant_id"], booked["reservation_code"]


def test_note_is_appended_only_for_the_owner():
    _, code = book_at_10_to_22()
    assert call(tools.update_reservation, code, "Someone Else", add_note="gluten allergy")["status"] == "error"
    first = call(tools.update_reservation, code, "alex", add_note="gluten allergy")
    assert first["notes"] == "gluten allergy" and first["updated"] == ["notes"]
    second = call(tools.update_reservation, code, "Alex", add_note="high chair")
    assert second["notes"] == "gluten allergy; high chair"
    assert call(tools.update_reservation, code, "Alex")["status"] == "error"  # nothing to change


def test_added_name_keeps_the_original_and_the_change_is_recorded():
    _, code = book_at_10_to_22(name="Ivan Petrov")
    assert call(tools.update_reservation, code, "Maria", add_name="Maria")["status"] == "error"  # not on it yet
    added = call(tools.update_reservation, code, "Ivan Petrov", add_name="Мария Георгиева", time="20:00")
    assert added["added_name"] == "Мария Георгиева" and added["updated"] == ["time", "names"]
    for name in ("Ivan Petrov", "мария георгиева"):                     # both names manage the booking now
        assert call(tools.update_reservation, code, name, add_note="window seat")["status"] == "success", name
    assert "added_name" not in call(tools.update_reservation, code, "Ivan", add_name="Ivan Petrov")  # already on it
    staff = next(r for r in tools.list_reservations() if r["code"] == code)
    assert staff["customer_name"] == "Ivan Petrov" and staff["extra_names"] == ["Мария Георгиева"]
    assert [h["change"] for h in staff["history"]] == [
        "time: 19:00 -> 20:00", "name added: Мария Георгиева", "note added: window seat", "note added: window seat"]
    assert "Мария" not in json.dumps(call(tools.find_reservation, code), ensure_ascii=False)  # not revealed by code


def test_callback_request_needs_a_name_and_a_way_to_reach_the_guest():
    ok = call(tools.request_callback, "event", "Иван Петров", "+359 888 123 456", "Детско парти за 25 деца в събота",
              date="2026-11-28", guests=25, city="Sofia")
    assert ok["status"] == "success" and ok["request_number"] == 1
    saved = call(tools.request_callback, "other", "Ana", "ana@example.com", "Wants a private room")
    assert saved["status"] == "success"
    for contact in ("", "call me", "123", "ana@example", "0888 123 456 789 012 345"):
        assert "contact" in call(tools.request_callback, "event", "Ana", contact, "A party")["message"], contact
    assert call(tools.request_callback, "party", "Ana", "0888123456", "A party")["status"] == "error"   # unknown kind
    assert call(tools.request_callback, "event", " ", "0888123456", "A party")["status"] == "error"     # no name
    assert call(tools.request_callback, "event", "Ana", "0888123456", "A party", date="2026-11-01")["status"] == "error"
    saved = tools.list_requests()
    assert [(q["customer_name"], q["kind"], q["status"]) for q in saved] == [("Ana", "other", "open"),
                                                                             ("Иван Петров", "event", "open")]
    assert saved[1]["guests"] == 25 and saved[1]["date"] == "2026-11-28"


def test_moving_a_reservation_checks_capacity_without_counting_itself():
    rid, code = book_at_10_to_22(guests=20)          # the restaurant is full at 19:00
    moved = call(tools.update_reservation, code, "Alex", time="19:30")  # overlaps only with itself
    assert moved["status"] == "success" and moved["time"] == "19:30" and moved["updated"] == ["time"]

    call(tools.book_table, rid, "Other", TOMORROW, "12:00", 20)
    blocked = call(tools.update_reservation, code, "Alex", time="12:30")
    assert blocked["status"] == "unavailable" and blocked["alternatives"]
    assert call(tools.find_reservation, code)["reservation"]["time"] == "19:30"  # unchanged

    assert "open" in call(tools.update_reservation, code, "Alex", time="23:00")["message"]


def test_cancelled_reservation_cannot_be_changed():
    _, code = book_at_10_to_22()
    call(tools.cancel_reservation, code, "Alex")
    assert "cancelled" in call(tools.update_reservation, code, "Alex", guests=3)["message"]


def test_search_by_part_of_the_name():
    result = call(tools.search_restaurants, name="marble hearth")
    assert [r["name"] for r in result["restaurants"]] == ["Marble Hearth"]



# --- Menus in the guest's language and prices -------------------------------------------

def test_menu_dish_names_follow_the_guest_language_and_prices_have_two_decimals():
    rid = restaurant("cuisine = 'Bulgarian'")["restaurant_id"]
    english = call(tools.get_menu, rid)
    bulgarian = call(tools.get_menu, rid, language="bg")
    assert [d["price"] for d in english["dishes"]] == [d["price"] for d in bulgarian["dishes"]]
    assert all(re.fullmatch(r"\d+\.\d\d EUR", d["price"]) for d in bulgarian["dishes"])
    assert all(re.search(r"[А-я]", d["dish"]) for d in bulgarian["dishes"])  # Cyrillic names
    assert call(tools.get_menu, rid, language="xx")["language"] == "en"  # unknown -> English


def test_bulgarian_menus_include_the_classics():
    names = {json.loads(n)["bg"] for (n,) in data.sqlite3.connect(config.DB_PATH).execute(
        "SELECT names FROM menu_items m JOIN restaurants r USING (restaurant_id) WHERE r.cuisine = 'Bulgarian'")}
    assert {"Телешко варено", "Пилешка супа", "Шкембе чорба", "Мусака", "Сарми", "Таратор"} <= names


def test_search_shows_price_per_person_and_can_sort_by_it():
    result = call(tools.search_restaurants, city="Sofia", cuisine="Italian", sort_by="price", limit=10)
    prices = [int(r["price_per_person"].split()[1]) for r in result["restaurants"]]
    assert prices == sorted(prices)
    assert all("price_range" not in r for r in result["restaurants"])  # no "$$$" for the agent to read out
    cheap = call(tools.search_restaurants, city="Sofia", max_price_per_person=15, limit=10)["restaurants"]
    assert cheap and all(int(r["price_per_person"].split()[1]) <= 15 for r in cheap)
    assert call(tools.search_restaurants, sort_by="cheapest")["status"] == "error"


def test_price_per_person_is_two_courses_at_the_median_rounded_to_5():
    assert data.price_per_person([4.0, 6.0, 8.0, 20.0]) == 15   # median 7 -> 14 -> 15
    assert data.price_per_person([2.0, 2.0]) == 5               # never below 5


def test_allergen_names_follow_the_guest_language():
    with_celery = "restaurant_id IN (SELECT restaurant_id FROM menu_items WHERE allergens LIKE '%celery%')"
    rid = restaurant(with_celery)["restaurant_id"]
    bulgarian = call(tools.get_menu, rid, language="bg", avoid_allergens=["celery"])
    named = {a for d in bulgarian["dishes"] for a in d["allergens"]}
    assert "целина" in named and "celery" not in named          # not the non-word "селера"
    flagged = [d for d in bulgarian["dishes"] if d["contains_avoided"]]
    assert flagged and all(d["contains_avoided"] == ["целина"] or "целина" in d["contains_avoided"] for d in flagged)
    assert call(tools.get_menu, rid, language="de")["dishes"][0]["allergens"][0] in data.ALLERGEN_NAMES["de"].values() \
        or call(tools.get_menu, rid, language="de")["dishes"][0]["allergens"] == ["keine angegeben"]
