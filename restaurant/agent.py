"""The agent: the tool-calling loop, the booking confirmation gate and history trimming.

The system prompt is in prompts.py."""
import concurrent.futures
import json

from openai import OpenAI

from . import config, decisions
from .decide import Action
from .prompts import build_system_prompt
from .registry import execute_tool_call
from .tool_schemas import TOOLS_SCHEMA

llm_client = OpenAI(base_url=config.LLM_BASE_URL, api_key="lm-studio", timeout=config.LLM_TIMEOUT_SECONDS)
_gate_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="wants-human")

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


BOOKING_UNCLEAR = json.dumps({
    "status": "error",
    "message": "Not booked: the guest's reply is not a clear yes to these exact details. Answer any question they "
               "asked, read the booking back and ask one explicit yes/no question; book in the next turn after a "
               "clear yes.",
})

BOOKING_NOT_AGREED = json.dumps({
    "status": "error",
    "message": "Not booked: the guest did not agree to this booking. Do not book it unless they ask for it again.",
})


def booking_gate(raw_arguments: str, guest_text: str, model: str | None, decisions_log: list | None) -> str | None:
    """The second key for book_table: None = book; otherwise the result to return instead."""
    if not config.DECISION_GATE:
        return None
    try:
        arguments = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        arguments = {}
    booking = (f"restaurant {arguments.get('restaurant_id')}, {arguments.get('date')} at {arguments.get('time')}, "
               f"{arguments.get('guests')} guests, name {arguments.get('customer_name')}")
    agree, instruction = decisions.decider(model).ask_many_sync(
        [decisions.CONFIRM_BOOKING, decisions.INSTRUCTION], guest_text, {"booking": booking})
    action = decisions.booking_action(agree, instruction)
    if decisions_log is not None:
        decisions_log += [{**d.record.model_dump(mode="json"), "policy_action": action.value}
                          for d in (agree, instruction)]
    if action == Action.ACT:
        return None
    return BOOKING_UNCLEAR if action == Action.CONFIRM else BOOKING_NOT_AGREED


DUPLICATE_CALL = json.dumps({
    "status": "duplicate",
    "message": "This exact call was already made in this turn and was not run again; use its result above.",
})


def call_key(name: str, raw_arguments: str):
    """Identifies a call by its tool and arguments, whatever the order of the keys."""
    try:
        return name, json.dumps(json.loads(raw_arguments or "{}"), sort_keys=True)
    except json.JSONDecodeError:
        return name, raw_arguments


def run_llm_turn(turn_history: list, tool_calls: list, spoken_language: str = None, model: str = None,
                 decisions_log: list = None):
    """Call the model, executing tool calls until it answers (or MAX_TOOL_ROUNDS is hit).

    Mutates turn_history and tool_calls; returns the final model message.

    Guard in code, not only in the prompt: book_table runs only for details that check_availability offered
    in an EARLIER turn, i.e. the guest has heard them and answered since. (In evals Gemma sometimes booked
    straight away when the guest said "Book a table ...".)
    An identical call repeated in the same turn is not run again (in evals Qwen sent the same book_table call
    twice in one round, which booked two tables).
    Then a calibrated booking gate (decisions.py) checks that the guest's latest message really agrees.
    In parallel, a "wants a person" gate: if the guest asked for one and no request_callback ran in this turn, the
    model is called once more with an instruction to start the handover steps.
    """
    last_user = max((i for i, m in enumerate(turn_history) if m.get("role") == "user"), default=0)
    guest_text = str(turn_history[last_user].get("content") or "") if turn_history else ""
    offered_before = confirmed_offers(turn_history[:last_user])
    done_this_turn = set()
    wants_human = start_wants_human_gate(guest_text, turn_history[:last_user], model)
    hint = ""
    msg = None
    calls_left = config.MAX_TOOL_ROUNDS + 1
    while calls_left > 0:
        calls_left -= 1
        response = llm_client.chat.completions.create(
            model=model or config.MODEL_NAME,
            messages=[{"role": "system", "content": build_system_prompt(spoken_language) + hint}] + turn_history,
            tools=TOOLS_SCHEMA,
        )
        msg = response.choices[0].message
        if not msg.tool_calls:
            if wants_human is not None and not hint and needs_handover_step(wants_human, tool_calls, decisions_log):
                hint, calls_left = decisions.HANDOVER_HINT, calls_left + 1   # one more call, this answer is dropped
                continue
            break
        turn_history.append(assistant_message_to_dict(msg))
        for tc in msg.tool_calls:
            key = call_key(tc.function.name, tc.function.arguments)
            if key in done_this_turn:
                output = DUPLICATE_CALL
            elif tc.function.name == "book_table" and not booking_is_confirmed(tc.function.arguments, offered_before):
                output = BOOK_WITHOUT_CONFIRMATION
            elif tc.function.name == "book_table" and (refusal := booking_gate(
                    tc.function.arguments, guest_text, model, decisions_log)):
                output = refusal
                done_this_turn.add(key)
            else:
                output = execute_tool_call(tc.function.name, tc.function.arguments)
                done_this_turn.add(key)
            tool_calls.append({"name": tc.function.name, "arguments": tc.function.arguments or "{}", "result": output})
            turn_history.append({"role": "tool", "tool_call_id": tc.id, "content": output})
    return msg


def start_wants_human_gate(guest_text: str, earlier: list, model: str | None):
    """Ask "does the guest ask for a person?" in a worker thread while the agent answers. None when off."""
    if not config.DECISION_GATE or not guest_text:
        return None
    context = {"previous": decisions.previous_assistant_message(earlier)}
    return _gate_pool.submit(decisions.decider(model).ask_sync, decisions.WANTS_HUMAN, guest_text, context)


def needs_handover_step(future, tool_calls: list, decisions_log: list | None) -> bool:
    """The guest asked for a person and the agent did not pass a callback on in this turn."""
    decision = future.result()
    called_back = any(c["name"] == "request_callback" for c in tool_calls)
    nudge = decisions.wants_human(decision) and not called_back
    if decisions_log is not None:
        decisions_log.append({**decision.record.model_dump(mode="json"), "tool": "request_callback",
                              "policy_action": "handover_step" if nudge else "none"})
    return nudge


def trim_history(history: list, max_turns: int = config.MAX_HISTORY_TURNS) -> list:
    """Keep the last `max_turns` user turns. Cuts only before a user message, so tool-call pairs stay complete."""
    user_indexes = [i for i, m in enumerate(history) if m.get("role") == "user"]
    if len(user_indexes) <= max_turns:
        return history
    return history[user_indexes[-max_turns]:]
