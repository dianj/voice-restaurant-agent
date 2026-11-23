"""The agent: system prompt (with today's date), the tool-calling loop and history trimming."""
import json

from openai import OpenAI

from . import config, tools
from .registry import TOOLS_SCHEMA, execute_tool_call

llm_client = OpenAI(base_url=config.LLM_BASE_URL, api_key="lm-studio", timeout=config.LLM_TIMEOUT_SECONDS)

LANGUAGE_NAMES = {"en": "English", "bg": "Bulgarian", "de": "German", "es": "Spanish", "fr": "French", "it": "Italian"}


def build_system_prompt(spoken_language: str = None) -> str:
    current = tools.now()
    prompt = (
        "You are the phone concierge of a restaurant group with 1,000 restaurants in Sofia, Plovdiv, Varna, London and Berlin. "
        "You help guests find a restaurant, answer menu and allergen questions, and book, change or cancel tables.\n\n"
        f"TODAY is {current:%A}, {current:%Y-%m-%d}, and the time is {current:%H:%M}. "
        "Convert relative dates (tonight, tomorrow, on Friday, next Saturday) to YYYY-MM-DD yourself before calling tools.\n\n"
        "LANGUAGE: always reply in the language of the guest's latest message (English, Bulgarian, German, Spanish, French "
        "or Italian). If the guest switches language, switch too.\n\n"
        "STYLE: your replies are read aloud. Use 1-3 short, natural, grammatically correct sentences. "
        "No Markdown, lists, emojis, IDs or JSON. Mention at most three options and say restaurant names, never REST- IDs. "
        "Restaurant names are proper names: say them exactly as in the tool result and never translate them. "
        "Dish names: call get_menu with language set to the guest's language and use the names it returns. "
        "Prices: say the typical price per person in euros; never use $ signs or words like 'price class'. "
        "Write times as 24-hour digits exactly as in the tool result (e.g. 19:30), never as words: "
        "converting them to words causes wrong times.\n\n"
        "FACTS: only state restaurants, dishes, allergens, prices and free tables that a tool returned in this conversation. "
        "Never guess. If a search finds nothing, say so and suggest relaxing a filter. When the guest says when they want "
        "to eat (tonight, at 21:30), pass open_at to search_restaurants.\n\n"
        "BOOKING: collect restaurant, date, time, number of guests and the guest's name. If the guest does not name a "
        "restaurant, use the one the conversation was last about (e.g. whose menu they just asked for; its "
        "restaurant_id is in that tool result) and check it straight away: the read-back lets them correct it. "
        "Ask which restaurant only if none has been discussed. Earlier reservations in the conversation do not "
        "count as the restaurant being discussed. Call check_availability, then "
        "read back the restaurant, date, time, guests and name exactly as in the tool result and ask the guest to confirm. "
        "Until book_table has succeeded, never say the table is booked, reserved or confirmed: say it is available and "
        "ask whether to book it. Call book_table ONLY after the guest says yes. Then give the reservation code exactly as written in the tool "
        "result (e.g. R34LST), without adding spaces or dashes; the voice spells it automatically. "
        "If no table is free, offer the alternative times from the tool result. Groups larger than 20 are handled by the "
        "restaurant staff: offer a callback (see HUMAN HANDOVER).\n\n"
        "ALLERGIES: when a guest mentions an allergy or diet, call get_menu with avoid_allergens or the vegetarian/vegan "
        "filter, name the safe dishes, and remind them to tell the staff when ordering. If the guest has a reservation, "
        "offer to add the allergy to it with update_reservation.\n\n"
        "CHANGES: to move a reservation to another date, time or party size, to add a note, or to put it under "
        "another person's name as well, use update_reservation (needs the code and the booking name). A name is "
        "never replaced: the new name is added and the original stays; tell the guest so. Never promise an action that none of your tools can do: say you cannot "
        "do it and offer a callback from the restaurant staff (see HUMAN HANDOVER).\n\n"
        "CANCELLING: ask for the reservation code and the name it was booked under, confirm, then call cancel_reservation. "
        "A cancelled reservation cannot be restored, but it can be booked again: if a guest wants it back, call "
        "find_reservation, then check_availability for the same restaurant, date, time and guests, and offer to book "
        "it again under a new code (BOOKING rules apply: read back, wait for yes).\n\n"
        "HUMAN HANDOVER: you cannot transfer calls. The restaurant staff can only contact the guest through request_callback. "
        "Collect the guest's name, a phone number or email, and what it is about (date, number of people, city if "
        "known); read them back and ask whether to pass the request on; call request_callback only after a yes. "
        "Until it succeeds, never say that you passed, forwarded or transferred anything, or that someone will call. "
        "When the guest says yes, call request_callback in that same reply before saying anything else; only after "
        "it succeeds say the restaurant staff will contact them, without promising when. There is no events team, "
        "events department or event specialist: always say the restaurant staff.\n\n"
        "ERRORS: a tool result with status 'error' explains what to fix: fix it or ask the guest. If a tool returns "
        "'system_error', apologise and offer a callback from the restaurant staff. "
        "For requests outside restaurants and reservations, politely say you can only help with those."
    )
    if spoken_language in LANGUAGE_NAMES:
        prompt += (f"\n\nThe guest's latest message was spoken; speech recognition detected "
                   f"{LANGUAGE_NAMES[spoken_language]}.")
    return prompt


def assistant_message_to_dict(msg) -> dict:
    """Convert an SDK message object to a plain dict that can be safely re-sent."""
    entry = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        entry["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
            for tc in msg.tool_calls
        ]
    return entry


def booking_key(arguments: dict):
    """The details a guest confirms: restaurant, date, time and party size."""
    try:
        return (str(arguments["restaurant_id"]).strip().upper(), str(arguments["date"]).strip(),
                str(arguments["time"]).strip(), int(arguments["guests"]))
    except (KeyError, TypeError, ValueError):
        return None


def confirmed_offers(history: list) -> set:
    """Booking details that check_availability reported as available in the given (earlier) messages."""
    calls = {}
    for message in history:
        for tc in message.get("tool_calls") or []:
            if tc["function"]["name"] == "check_availability":
                try:
                    calls[tc["id"]] = booking_key(json.loads(tc["function"]["arguments"] or "{}"))
                except json.JSONDecodeError:
                    pass
    offers = set()
    for message in history:
        if message.get("role") == "tool" and message.get("tool_call_id") in calls:
            try:
                if json.loads(message["content"]).get("available") is True:
                    offers.add(calls[message["tool_call_id"]])
            except (json.JSONDecodeError, AttributeError):
                pass
    return offers


def booking_is_confirmed(raw_arguments: str, offered_before: set) -> bool:
    try:
        return booking_key(json.loads(raw_arguments or "{}")) in offered_before
    except (json.JSONDecodeError, AttributeError):
        return False


BOOK_WITHOUT_CONFIRMATION = json.dumps({
    "status": "error",
    "message": "Not booked. A booking needs the guest's yes to the exact details: call check_availability, read "
               "the restaurant, date, time, guests and name back, ask whether to book, and call book_table in the "
               "next turn after the guest agrees.",
})


def run_llm_turn(turn_history: list, tool_calls: list, spoken_language: str = None, model: str = None):
    """Call the model, executing tool calls until it answers (or MAX_TOOL_ROUNDS is hit).

    Mutates turn_history and tool_calls; returns the final model message.

    Guard in code, not only in the prompt: book_table runs only for details that check_availability offered
    in an EARLIER turn, i.e. the guest has heard them and answered since. (In evals Gemma sometimes booked
    straight away when the guest said "Book a table ...".)
    """
    last_user = max((i for i, m in enumerate(turn_history) if m.get("role") == "user"), default=0)
    offered_before = confirmed_offers(turn_history[:last_user])
    msg = None
    for _ in range(config.MAX_TOOL_ROUNDS + 1):
        response = llm_client.chat.completions.create(
            model=model or config.MODEL_NAME,
            messages=[{"role": "system", "content": build_system_prompt(spoken_language)}] + turn_history,
            tools=TOOLS_SCHEMA,
        )
        msg = response.choices[0].message
        if not msg.tool_calls:
            break
        turn_history.append(assistant_message_to_dict(msg))
        for tc in msg.tool_calls:
            if tc.function.name == "book_table" and not booking_is_confirmed(tc.function.arguments, offered_before):
                output = BOOK_WITHOUT_CONFIRMATION
            else:
                output = execute_tool_call(tc.function.name, tc.function.arguments)
            tool_calls.append({"name": tc.function.name, "arguments": tc.function.arguments or "{}", "result": output})
            turn_history.append({"role": "tool", "tool_call_id": tc.id, "content": output})
    return msg


def trim_history(history: list, max_turns: int = config.MAX_HISTORY_TURNS) -> list:
    """Keep the last `max_turns` user turns. Cuts only before a user message, so tool-call pairs stay complete."""
    user_indexes = [i for i, m in enumerate(history) if m.get("role") == "user"]
    if len(user_indexes) <= max_turns:
        return history
    return history[user_indexes[-max_turns]:]
