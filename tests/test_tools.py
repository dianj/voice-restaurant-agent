"""Unit tests for the booking rules, search and menu tools (no LLM). The clock is frozen at Fri 2026-11-20 10:00."""
import json

from restaurant import config, data, tools


def call(fn, *args, **kwargs):
    return json.loads(fn(*args, **kwargs))


def restaurant(where: str = "1 = 1", params=()):
    with tools.db() as conn:
        return dict(conn.execute(f"SELECT * FROM restaurants WHERE {where} ORDER BY restaurant_id LIMIT 1", params).fetchone())


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
        (TOMORROW, "19:00", 25): "events team",
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
    assert booking["status"] == "success" and booking["reservation_code"].startswith("R-")
    code = booking["reservation_code"]

    found = call(tools.find_reservation, code.lower().replace("-", ""))  # spoken/typed variants
    assert found["reservation"]["customer_name"] == "Alex Ivanov"
    assert found["reservation"]["notes"] == "gluten allergy"

    assert call(tools.cancel_reservation, code, "Someone Else")["status"] == "error"
    assert call(tools.cancel_reservation, code, "alex ivanov")["cancelled"] is True
    assert call(tools.find_reservation, code)["reservation"]["status"] == "cancelled"


def test_cancelled_reservation_frees_the_seats():
    r = restaurant("seating_capacity = 20 AND opening_hours = '10:00-22:00'")
    rid = r["restaurant_id"]
    code = call(tools.book_table, rid, "Group", TOMORROW, "19:00", 20)["reservation_code"]
    assert call(tools.check_availability, rid, TOMORROW, "19:00", 1)["available"] is False
    call(tools.cancel_reservation, code, "Group")
    assert call(tools.check_availability, rid, TOMORROW, "19:00", 20)["available"] is True


def test_booking_requires_a_name():
    assert "name" in call(tools.book_table, "REST-0001", " ", TOMORROW, "19:00", 2)["message"]
