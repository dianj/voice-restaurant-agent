"""The tool schemas sent to the model and the dispatcher that runs the model's tool calls."""
import json
import logging

from .data import ALLERGENS, CUISINES, LOCATIONS
from .tools import book_table, cancel_reservation, check_availability, find_reservation, get_menu, search_restaurants

AREAS = sorted({area for areas in LOCATIONS.values() for area in areas})

TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "search_restaurants",
        "description": "Find restaurants in the network by cuisine, city, area, minimum rating and/or a time they are open. "
                       "Results are sorted by rating. Use English names for cuisine, city and area.",
        "parameters": {"type": "object", "properties": {
            "cuisine": {"type": "string", "enum": CUISINES},
            "city": {"type": "string", "enum": list(LOCATIONS)},
            "area": {"type": "string", "enum": AREAS, "description": "Neighbourhood inside the city."},
            "min_rating": {"type": "number", "description": "0-5, e.g. 4.5"},
            "open_at": {"type": "string", "description": "HH:MM; only restaurants open at that time."},
            "limit": {"type": "integer", "description": "1-10, default 5."},
        }},
    }},
    {"type": "function", "function": {
        "name": "get_menu",
        "description": "Menu of one restaurant with ingredients, declared allergens, vegetarian/vegan flags and prices. "
                       "Pass avoid_allergens to get the dishes that are safe for the guest.",
        "parameters": {"type": "object", "properties": {
            "restaurant_id": {"type": "string", "description": "e.g. REST-0042 (from search_restaurants)."},
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
        "description": "Look up a reservation by its code (format R-XXXXX).",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
    }},
    {"type": "function", "function": {
        "name": "cancel_reservation",
        "description": "Cancel a reservation. Needs the code and the name it was booked under. Confirm with the guest first.",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string"}, "customer_name": {"type": "string"},
        }, "required": ["code", "customer_name"]},
    }},
]

AVAILABLE_FUNCTIONS = {
    "search_restaurants": search_restaurants,
    "get_menu": get_menu,
    "check_availability": check_availability,
    "book_table": book_table,
    "find_reservation": find_reservation,
    "cancel_reservation": cancel_reservation,
}


def execute_tool_call(fn_name: str, raw_arguments: str) -> str:
    """Run one model-requested tool call; always return a JSON string, never raise."""
    if fn_name not in AVAILABLE_FUNCTIONS:
        return json.dumps({"status": "error", "message": f"Unknown tool '{fn_name}'. Available: {list(AVAILABLE_FUNCTIONS)}"})
    try:
        fn_args = json.loads(raw_arguments) if raw_arguments else {}
    except json.JSONDecodeError as e:
        return json.dumps({"status": "error", "message": f"Tool arguments are not valid JSON: {e}"})
    if not isinstance(fn_args, dict):
        return json.dumps({"status": "error", "message": "Tool arguments must be a JSON object."})
    try:
        return AVAILABLE_FUNCTIONS[fn_name](**fn_args)
    except TypeError as e:
        return json.dumps({"status": "error", "message": f"Invalid arguments for '{fn_name}': {e}"})
    except Exception:
        # Internal details stay in the server log; the model only learns that the system failed.
        logging.getLogger("restaurant").exception("Tool %s failed", fn_name)
        return json.dumps({"status": "system_error", "message": "The booking system is temporarily unavailable."})
