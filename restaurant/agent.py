"""The agent: the tool-calling loop, the booking confirmation gate and history trimming.

The system prompt is in prompts.py."""
import concurrent.futures
import json
import re

from openai import OpenAI

from . import config, decisions
from .decide import Action
from .prompts import build_system_prompt
from .registry import execute_tool_call
from .tool_schemas import TOOLS_SCHEMA
from .tools import name_key, name_parts, normalize_code

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


def last_reply_index(history: list) -> int | None:
    """Index of the assistant's last reply the guest heard (text, not a tool round)."""
    for i in range(len(history) - 1, -1, -1):
        message = history[i]
        if message.get("role") == "assistant" and message.get("content") and not message.get("tool_calls"):
            return i
    return None


def tool_results(history: list) -> list:
    """(index of the result, tool name, arguments, result) for every tool call in the history, in order."""
    calls, out = {}, []
    for i, message in enumerate(history):
        for tc in message.get("tool_calls") or []:
            try:
                arguments = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls[tc["id"]] = (tc["function"]["name"], arguments if isinstance(arguments, dict) else {})
        if message.get("role") == "tool" and message.get("tool_call_id") in calls:
            try:
                outcome = json.loads(message["content"])
            except (json.JSONDecodeError, TypeError):
                outcome = {}
            name, arguments = calls[message["tool_call_id"]]
            out.append((i, name, arguments, outcome if isinstance(outcome, dict) else {}))
    return out


def offered_before(earlier: list) -> set:
    """The booking the guest can say yes to now: the last check_availability before the assistant's last reply, if
    the table was available and that reply read its time back. An older offer does not count after the
    conversation moved on: a yes to "Shall I email you the menu?" is not a yes to it (review T22.A2)."""
    last = last_reply_index(earlier)
    checks = [(arguments, outcome) for _, name, arguments, outcome in tool_results(earlier[:last or 0])
              if name == "check_availability"]
    if last is None or not checks:
        return set()
    arguments, outcome = checks[-1]
    key = booking_key(arguments)
    if key is None or outcome.get("available") is not True or str(outcome.get("time", key[2])) not in \
            earlier[last]["content"]:
        return set()
    return {key}


def found_reservation(earlier: list, code) -> dict | None:
    """The reservation find_reservation returned for this code in an earlier turn (before the assistant's last
    reply), or None: a cancel needs the guest to have heard which booking it is (review T22.A3)."""
    target, last = normalize_code(code), last_reply_index(earlier)
    for i, name, arguments, outcome in reversed(tool_results(earlier)):
        if name == "find_reservation" and outcome.get("status") == "success" \
                and normalize_code(arguments.get("code")) == target:
            return outcome.get("reservation") if last is not None and last > i else None
    return None


def contact_read_back(earlier: list, arguments: dict) -> bool:
    """The assistant's last reply read the guest's (first) name and phone number or email back (before
    request_callback). The first name is enough: "Разбрах, Иван. ... Телефонът ви е 0888 123 456" is a read-back."""
    last = last_reply_index(earlier)
    if last is None:
        return False
    reply = earlier[last]["content"]
    parts = name_parts(arguments.get("customer_name"))
    name = parts[0] if parts else ""
    contact = str(arguments.get("contact") or "").strip()
    digits = re.sub(r"\D", "", contact)
    if "@" in contact:
        contact_ok = contact.lower() in reply.lower()
    else:
        contact_ok = len(digits) >= 7 and digits in re.sub(r"\D", "", reply)
    return bool(name) and name in name_key(reply) and contact_ok


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

CANCEL_WITHOUT_CONFIRMATION = json.dumps({
    "status": "error",
    "message": "Not cancelled. First call find_reservation for this code, read the restaurant, date and time back "
               "and ask whether to cancel it; call cancel_reservation in the next turn, after the guest says yes.",
})

CALLBACK_WITHOUT_READ_BACK = json.dumps({
    "status": "error",
    "message": "Not passed on. Read the guest's name, phone number or email and what it is about back to them and "
               "ask whether to pass the request on; call request_callback in the next turn, after the guest says yes. "
               "Do not say it was passed on or that anyone will call: it was not.",
})

STEP_UNCLEAR = json.dumps({
    "status": "error",
    "message": "Not done: the guest's reply is not a clear yes to this. Answer any question they asked, read the "
               "details back and ask one explicit yes/no question; do it in the next turn after a clear yes.",
})

STEP_NOT_AGREED = json.dumps({
    "status": "error",
    "message": "Not done: the guest did not agree. Do not do it unless they ask for it again.",
})


def ask_gate(question, guest_text: str, context: dict, model: str | None, decisions_log: list | None) -> Action:
    """A yes/no gate and the "text addressed to the system" check, in parallel; the policy's action."""
    agree, instruction = decisions.decider(model).ask_many_sync([question, decisions.INSTRUCTION], guest_text, context)
    action = decisions.booking_action(agree, instruction, question)
    if decisions_log is not None:
        decisions_log += [{**d.record.model_dump(mode="json"), "policy_action": action.value}
                          for d in (agree, instruction)]
    return action


def parse_arguments(raw_arguments: str) -> dict:
    try:
        arguments = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return {}
    return arguments if isinstance(arguments, dict) else {}


def booking_gate(raw_arguments: str, guest_text: str, model: str | None, decisions_log: list | None,
                 previous: str = "") -> str | None:
    """The second key for book_table: None = book; otherwise the result to return instead."""
    if not config.DECISION_GATE:
        return None
    arguments = parse_arguments(raw_arguments)
    booking = (f"restaurant {arguments.get('restaurant_id')}, {arguments.get('date')} at {arguments.get('time')}, "
               f"{arguments.get('guests')} guests, name {arguments.get('customer_name')}")
    action = ask_gate(decisions.CONFIRM_BOOKING, guest_text, {"booking": booking, "previous": previous}, model,
                      decisions_log)
    if action == Action.ACT:
        return None
    return BOOKING_UNCLEAR if action == Action.CONFIRM else BOOKING_NOT_AGREED


def step_gate(step: str, guest_text: str, model: str | None, decisions_log: list | None, previous: str) -> str | None:
    """The yes gate for a cancel or a callback request: None = do it; otherwise the result to return instead."""
    if not config.DECISION_GATE:
        return None
    action = ask_gate(decisions.CONFIRM_STEP, guest_text, {"step": step, "previous": previous}, model, decisions_log)
    if action == Action.ACT:
        return None
    return STEP_UNCLEAR if action == Action.CONFIRM else STEP_NOT_AGREED


def guard(name: str, raw_arguments: str, earlier: list, offered: set, guest_text: str, model: str | None,
          decisions_log: list | None) -> str | None:
    """Checks in code (and the calibrated gates) before a tool that books, cancels or passes contact data on.
    None = run the tool; otherwise the result to return to the model instead."""
    previous = decisions.previous_assistant_message(earlier)
    if name == "book_table":
        if not booking_is_confirmed(raw_arguments, offered):
            return BOOK_WITHOUT_CONFIRMATION
        return booking_gate(raw_arguments, guest_text, model, decisions_log, previous)
    arguments = parse_arguments(raw_arguments)
    if name == "cancel_reservation":
        found = found_reservation(earlier, arguments.get("code"))
        if found is None:
            return CANCEL_WITHOUT_CONFIRMATION
        step = (f"cancel reservation {normalize_code(arguments.get('code'))} at {found.get('restaurant')}, "
                f"{found.get('date')} at {found.get('time')}")
        return step_gate(step, guest_text, model, decisions_log, previous)
    if name == "request_callback":
        if not contact_read_back(earlier, arguments):
            return CALLBACK_WITHOUT_READ_BACK
        step = (f"pass a callback request to the restaurant staff: name {arguments.get('customer_name')}, contact "
                f"{arguments.get('contact')}, about: {arguments.get('details')}")
        return step_gate(step, guest_text, model, decisions_log, previous)
    return None


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

    Guards in code, not only in the prompt (`guard`): book_table runs only for the details that the last
    check_availability offered before the assistant's last reply, and only if that reply read them back, i.e. the
    guest has heard them and answered since. (In evals Gemma sometimes booked straight away when the guest said
    "Book a table ...".) cancel_reservation runs only for a reservation find_reservation showed in an earlier turn,
    request_callback only after the assistant read the name and contact back; each then needs a yes from its gate.
    An identical call repeated in the same turn is not run again (in evals Qwen sent the same book_table call
    twice in one round, which booked two tables).
    Then a calibrated booking gate (decisions.py) checks that the guest's latest message really agrees.
    In parallel, a "wants a person" gate: if the guest asked for one and no request_callback ran in this turn, the
    model is called once more with an instruction to start the handover steps.
    """
    last_user = max((i for i, m in enumerate(turn_history) if m.get("role") == "user"), default=0)
    guest_text = str(turn_history[last_user].get("content") or "") if turn_history else ""
    earlier = turn_history[:last_user]
    offered = offered_before(earlier)
    done_this_turn = set()
    wants_human = start_wants_human_gate(guest_text, earlier, model)
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
            elif refusal := guard(tc.function.name, tc.function.arguments, earlier, offered, guest_text, model,
                                  decisions_log):
                output = refusal
                done_this_turn.add(key)
            else:
                output = execute_tool_call(tc.function.name, tc.function.arguments, spoken=spoken_language is not None)
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
