"""The tool schemas (names, descriptions, parameters) sent to the model with every request.

The descriptions are instructions the model reads, kept one sentence group per line; ruff's line-length check
is relaxed for this file only.
"""
from .data import ALLERGENS, CUISINES, LOCATIONS

AREAS = sorted({area for areas in LOCATIONS.values() for area in areas})

TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "search_restaurants",
        "description": "Find restaurants in the network by name, cuisine, city, area, minimum rating, budget and/or a time "
                       "they are open. Each result has a typical price per person in EUR. Sorted by rating unless "
                       "sort_by='price'. Use it to get the restaurant_id when the guest names a "
                       "restaurant. Use English names for cuisine, city and area.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Part of the restaurant name, e.g. 'Marble Hearth'."},
            "cuisine": {"type": "string", "enum": CUISINES},
            "city": {"type": "string", "enum": list(LOCATIONS)},
            "area": {"type": "string", "enum": AREAS, "description": "Neighbourhood inside the city."},
            "min_rating": {"type": "number", "description": "0-5, e.g. 4.5"},
            "open_at": {"type": "string", "description": "HH:MM. Use it whenever the guest mentions when they want to "
                                                       "eat (tonight -> 20:00, at 21:30 -> 21:30, now -> current time)."},
            "max_price_per_person": {"type": "integer", "description": "Budget in EUR per person, e.g. 25."},
            "sort_by": {"type": "string", "enum": ["rating", "price"],
                        "description": "rating (default) or price = cheapest first, e.g. for 'the cheapest one'."},
            "limit": {"type": "integer", "description": "1-10, default 5."},
        }},
    }},
    {"type": "function", "function": {
        "name": "get_menu",
        "description": "Menu of one restaurant with ingredients, declared allergens, vegetarian/vegan flags and prices. "
                       "Dish names come back in the requested language. Pass avoid_allergens to get the dishes that are "
                       "safe for the guest.",
        "parameters": {"type": "object", "properties": {
            "restaurant_id": {"type": "string", "description": "e.g. REST-0042 (from search_restaurants)."},
            "language": {"type": "string", "enum": ["en", "bg", "de", "es", "fr", "it"],
                         "description": "The guest's language, for the dish names."},
            "avoid_allergens": {"type": "array", "items": {"type": "string", "enum": list(ALLERGENS)}},
            "vegetarian_only": {"type": "boolean"},
            "vegan_only": {"type": "boolean"},
        }, "required": ["restaurant_id"]},
    }},
    {"type": "function", "function": {
        "name": "check_availability",
        "description": "Check if a table is free. Does NOT book. If unavailable, the result lists the nearest free times.",
        "parameters": {"type": "object", "properties": {
            "restaurant_id": {"type": "string"},
            "date": {"type": "string", "description": "YYYY-MM-DD"},
            "time": {"type": "string", "description": "HH:MM, 24-hour, on :00 or :30"},
            "guests": {"type": "integer"},
        }, "required": ["restaurant_id", "date", "time", "guests"]},
    }},
    {"type": "function", "function": {
        "name": "book_table",
        "description": "Create a confirmed reservation. Call it ONLY after the guest has confirmed restaurant, date, time, "
                       "number of guests and their name. Returns a reservation code to read back to the guest.",
        "parameters": {"type": "object", "properties": {
            "restaurant_id": {"type": "string"},
            "customer_name": {"type": "string"},
            "date": {"type": "string", "description": "YYYY-MM-DD"},
            "time": {"type": "string", "description": "HH:MM"},
            "guests": {"type": "integer"},
            "phone": {"type": "string"},
            "notes": {"type": "string", "description": "e.g. allergy, high chair, terrace preference."},
        }, "required": ["restaurant_id", "customer_name", "date", "time", "guests"]},
    }},
    {"type": "function", "function": {
        "name": "find_reservation",
        "description": "Look up a reservation by its code (e.g. R34LST). The booking name is not returned: "
                       "ask the guest for it before changing or cancelling.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
    }},
    {"type": "function", "function": {
        "name": "update_reservation",
        "description": "Change an existing reservation: add a note (allergy, high chair, occasion), add another "
                       "name, and/or move it to another date, time or number of guests. Needs the code and a name "
                       "already on the booking. Free seats are checked again; if the new slot is full, nothing "
                       "changes and alternatives are returned.",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string"}, "customer_name": {"type": "string"},
            "add_note": {"type": "string", "description": "Appended to the existing notes."},
            "date": {"type": "string", "description": "New date, YYYY-MM-DD."},
            "time": {"type": "string", "description": "New time, HH:MM."},
            "guests": {"type": "integer", "description": "New number of guests."},
            "add_name": {"type": "string", "description": "Another person the table is also under, e.g. a colleague "
                                                          "who will come instead. The original name stays on the "
                                                          "booking; a name cannot be removed or replaced."},
        }, "required": ["code", "customer_name"]},
    }},
    {"type": "function", "function": {
        "name": "request_callback",
        "description": "Hand the guest over to a human: save a request for the restaurant staff to contact them. Use it for "
                       "events and parties, groups over 20, and anything you cannot do yourself. Collect the name, "
                       "a phone number or email and what it is about, read them back, and call this only after the "
                       "guest says yes. Until it succeeds, never say the request was passed on.",
        "parameters": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["event", "large_group", "other"]},
            "customer_name": {"type": "string"},
            "contact": {"type": "string", "description": "Phone number or email address, as the guest gave it."},
            "details": {"type": "string", "description": "One or two sentences in the guest's language: what, where, when."},
            "date": {"type": "string", "description": "YYYY-MM-DD, if the guest gave one."},
            "guests": {"type": "integer", "description": "Number of people, if known."},
            "city": {"type": "string"},
        }, "required": ["kind", "customer_name", "contact", "details"]},
    }},
    {"type": "function", "function": {
        "name": "cancel_reservation",
        "description": "Cancel a reservation. Needs the code and the name it was booked under. Confirm with the guest first.",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string"}, "customer_name": {"type": "string"},
        }, "required": ["code", "customer_name"]},
    }},
]
