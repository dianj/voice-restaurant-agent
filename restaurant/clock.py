"""The one clock of the app: UTC inside, and each restaurant's city in its own time zone.

The restaurants are in three time zones (Sofia, Plovdiv and Varna share one), so "today", "tonight" and "is this
time in the past?" depend on the restaurant, not on the server. Stored timestamps are UTC with an offset.
Tests freeze `utc_now`.
"""
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

CITY_ZONES = {
    "Sofia": "Europe/Sofia",
    "Plovdiv": "Europe/Sofia",
    "Varna": "Europe/Sofia",
    "London": "Europe/London",
    "Berlin": "Europe/Berlin",
}
HOME_ZONE = "Europe/Sofia"     # the group's own time zone: for anything not tied to one restaurant


def utc_now() -> datetime:
    return datetime.now(UTC)


def zone_of(city: str | None) -> ZoneInfo:
    return ZoneInfo(CITY_ZONES.get(city or "", HOME_ZONE))


def now_in(city: str | None = None) -> datetime:
    """Wall-clock time in the city's time zone, without tzinfo (booking dates and times are local and naive)."""
    return utc_now().astimezone(zone_of(city)).replace(tzinfo=None)


def earliest_today() -> date:
    """The earliest calendar date any of the cities is on now (the westernmost zone after midnight elsewhere)."""
    return min(now_in(city).date() for city in CITY_ZONES)


def stamp() -> str:
    """A stored timestamp: UTC with its offset, e.g. 2026-11-20T08:00:00+00:00."""
    return utc_now().isoformat(timespec="seconds")


def local_times() -> list[tuple[list[str], datetime]]:
    """(cities, local now) per time zone, in the order the cities are listed."""
    groups: dict[str, list[str]] = {}
    for city, zone in CITY_ZONES.items():
        groups.setdefault(zone, []).append(city)
    return [(cities, now_in(cities[0])) for cities in groups.values()]


def sofia_to_utc(naive: datetime) -> datetime:
    """A naive Sofia wall-clock time (how tests and eval cases give "now") as an aware UTC instant."""
    return naive.replace(tzinfo=ZoneInfo(HOME_ZONE)).astimezone(UTC)
