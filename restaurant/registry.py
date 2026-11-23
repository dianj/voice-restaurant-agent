"""The dispatcher that runs the model's tool calls. The schemas sent to the model are in tool_schemas.py."""
import json
import logging

from .tools import (
    book_table,
    cancel_reservation,
    check_availability,
    find_reservation,
    get_menu,
    request_callback,
    search_restaurants,
    update_reservation,
)

AVAILABLE_FUNCTIONS = {
    "search_restaurants": search_restaurants,
    "get_menu": get_menu,
    "check_availability": check_availability,
    "book_table": book_table,
    "find_reservation": find_reservation,
    "update_reservation": update_reservation,
    "request_callback": request_callback,
    "cancel_reservation": cancel_reservation,
}


def execute_tool_call(fn_name: str, raw_arguments: str) -> str:
    """Run one model-requested tool call; always return a JSON string, never raise."""
    if fn_name not in AVAILABLE_FUNCTIONS:
        return json.dumps({"status": "error",
                           "message": f"Unknown tool '{fn_name}'. Available: {list(AVAILABLE_FUNCTIONS)}"})
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
