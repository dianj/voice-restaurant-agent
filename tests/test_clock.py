"""One clock, five cities (review T22.A1, T25.S7): every booking rule uses the restaurant's own local time, the prompt
gives today per time zone, and stored timestamps are UTC with an offset."""
import json
import pathlib
import re
from datetime import UTC, datetime

from restaurant import clock, prompts, tools


def call(fn, *args, **kwargs):
    return json.loads(fn(*args, **kwargs))


def freeze(monkeypatch, iso_utc: str):
    instant = datetime.fromisoformat(iso_utc).replace(tzinfo=UTC)
    monkeypatch.setattr(clock, "utc_now", lambda: instant)


def open_at(city: str, hhmm: str):
    """A restaurant in `city` where a table at `hhmm` fits the opening hours and the last seating."""
    with tools.db() as conn:
        rows = conn.execute("SELECT * FROM restaurants WHERE city = ? ORDER BY restaurant_id", (city,)).fetchall()
    wanted = tools.minutes(hhmm)
    for row in rows:
        opens, closes = tools.opening_window(row["opening_hours"])
        if opens <= wanted <= closes - 90:
            return row["restaurant_id"]
    raise AssertionError(f"no restaurant in {city} takes a table at {hhmm}")


def test_at_19_in_sofia_a_london_table_at_18_is_not_in_the_past(monkeypatch):
    freeze(monkeypatch, "2026-11-25T17:00:00")          # 19:00 in Sofia, 17:00 in London
    london = call(tools.check_availability, open_at("London", "18:00"), "2026-11-25", "18:00", 2)
    assert london["status"] != "error", london
    sofia = call(tools.check_availability, open_at("Sofia", "18:00"), "2026-11-25", "18:00", 2)
    assert sofia["status"] == "error" and "past" in sofia["message"]


def test_after_midnight_in_sofia_it_is_still_the_25th_in_london(monkeypatch):
    freeze(monkeypatch, "2026-11-25T22:45:00")          # 00:45 on the 26th in Sofia, 22:45 on the 25th in London
    prompt = prompts.build_system_prompt()
    assert re.search(r"London: Wednesday, 2026-11-25, 22:45", prompt)
    assert re.search(r"Sofia, Plovdiv, Varna: Thursday, 2026-11-26, 00:45", prompt)
    assert re.search(r"Berlin: Wednesday, 2026-11-25, 23:45", prompt)
    args = dict(kind="event", customer_name="Ana Lee", contact="+44 20 7946 0000", details="A birthday party")
    assert call(tools.request_callback, **args, date="2026-11-25", city="London")["status"] == "success"
    assert call(tools.request_callback, **args, date="2026-11-25", city="Sofia")["status"] == "error"


def test_a_callback_without_a_city_accepts_any_date_that_is_today_somewhere(monkeypatch):
    freeze(monkeypatch, "2026-11-25T22:45:00")
    args = dict(kind="event", customer_name="Ana Lee", contact="+44 20 7946 0000", details="A birthday party")
    assert call(tools.request_callback, **args, date="2026-11-25")["status"] == "success"
    assert call(tools.request_callback, **args, date="2026-11-24")["status"] == "error"


def test_stored_timestamps_are_utc_with_an_offset(monkeypatch):
    freeze(monkeypatch, "2026-11-20T08:00:00")          # 10:00 in Sofia
    booked = call(tools.book_table, open_at("Sofia", "19:00"), "Maria", "2026-11-21", "19:00", 2)
    row = next(r for r in tools.list_reservations() if r["code"] == booked["reservation_code"])
    assert row["created_at"] == "2026-11-20T08:00:00+00:00"


def test_a_naive_sofia_time_maps_to_the_same_instant():
    """Tests and eval cases give 'now' as a naive Sofia wall-clock time."""
    assert clock.sofia_to_utc(datetime(2026, 11, 20, 10, 0)) == datetime(2026, 11, 20, 8, 0, tzinfo=UTC)
    assert clock.sofia_to_utc(datetime(2026, 7, 1, 10, 0)) == datetime(2026, 7, 1, 7, 0, tzinfo=UTC)   # summer time


def test_every_city_has_a_time_zone():
    from restaurant.data import LOCATIONS
    assert set(clock.CITY_ZONES) == set(LOCATIONS)


def test_no_naive_now_outside_the_clock_module():
    naive = re.compile(r"datetime\.now\(\s*\)|date\.today\(\)|datetime\.utcnow\(|\btime\.localtime\(")
    package = pathlib.Path(tools.__file__).parent
    offenders = [str(p.relative_to(package)) for p in package.rglob("*.py")
                 if p.name != "clock.py" and naive.search(p.read_text(encoding="utf-8"))]
    assert offenders == []
