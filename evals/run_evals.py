"""LLM evaluation: does the real model use the tools correctly and say the right thing?

Unlike the deterministic tests (which use a scripted fake LLM), this sends real conversations to the
model loaded in LM Studio. Each case starts from a fresh copy of the demo data with the clock frozen at
Friday 2026-11-20 10:00, and is graded on the *outcome* (reservations in the database, tool calls,
reply), not on exact wording.

Usage (LM Studio must be running; models are loaded one at a time, Gemma again at the end):
    python evals/run_evals.py --repeat 2                       # all models in ALLOWED_MODELS
    python evals/run_evals.py --models gemma-4-26b-a4b-it --cases book-after-confirmation time-as-digits
    python evals/run_evals.py --case-file other_cases.py --results-dir other/results --output baseline
Results: evals/results/latest.md (summary table) and a JSON file with every reply.
"""
import argparse
import importlib.util
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
_TMP = tempfile.mkdtemp(prefix="restaurant-evals-")
os.environ["RESTAURANT_DB_PATH"] = os.path.join(_TMP, "eval.db")

from restaurant import agent, config, data, lmstudio, tools  # noqa: E402
from restaurant.voice import detect_text_language  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evals", "results")
TEMPLATE_DB = os.path.join(_TMP, "template.db")
NOW = datetime(2026, 11, 20, 10, 0)  # Friday
TOMORROW = "2026-11-21"

# Restaurants from the deterministic demo data.
MARBLE = ("REST-0498", "Marble Hearth")      # Italian, Sofia Center, 17:00-23:00, 100 seats
OLIVE = ("REST-0586", "Olive Fork")          # Bulgarian, Sofia Lozenets, 10:00-22:00, 20 seats
GRAND = ("REST-0495", "Grand Courtyard")     # Japanese, Sofia Lozenets, 10:00-22:00, 20 seats


# --- helpers for checks ----------------------------------------------------------------

def calls(run, name):
    return [json.loads(c["arguments"] or "{}") for c in run["tools"] if c["name"] == name]


def booking_key_of(arguments):
    return (arguments.get("restaurant_id"), arguments.get("date"), arguments.get("time"), arguments.get("guests"))


def called(run, name):
    return bool(calls(run, name))


def cyrillic(text):
    return bool(re.search(r"[Ѐ-ӿ]", text))


def mentions(text, *words):
    lower = text.lower()
    return any(w in lower for w in words)


# A colleague really calls back (request_callback), but there is no events team, department or specialist.
INVENTED_TEAM = ("events team", "event team", "events department", "event specialist", "events specialist",
                 "event coordinator", "events coordinator", "event planner", "events manager", "event manager")


def no_invented_team(run):
    return not any(mentions(text, *INVENTED_TEAM) for text in run["lead_replies"] + [run["reply"]])


def reservations(run):
    return run["reservations"]


def one_booking(run, **expected):
    rows = [r for r in reservations(run) if r["status"] == "confirmed"]
    return len(rows) == 1 and all(str(rows[0][k]).lower() == str(v).lower() for k, v in expected.items())


def last_result(run, name):
    results = [json.loads(c["result"]) for c in run["tools"] if c["name"] == name]
    return results[-1] if results else {}


def unsafe_dishes(restaurant_id, allergen):
    with tools.db() as conn:
        rows = conn.execute("SELECT dish_name, allergens FROM menu_items WHERE restaurant_id = ?", (restaurant_id,)).fetchall()
    return [r["dish_name"] for r in rows if allergen in r["allergens"]]


def menu_names(restaurant_id, language):
    with tools.db() as conn:
        rows = conn.execute("SELECT names FROM menu_items WHERE restaurant_id = ?", (restaurant_id,)).fetchall()
    return [json.loads(r["names"])[language] for r in rows]


def cheapest(**filters):
    return json.loads(tools.search_restaurants(sort_by="price", limit=1, **filters))["restaurants"][0]["name"]


def book(name="Maria", restaurant=MARBLE, time_="19:00", guests=2):
    return json.loads(tools.book_table(restaurant[0], name, TOMORROW, time_, guests))["reservation_code"]


def seed_booking():
    return {"code": book()}


def seed_cancelled_booking():
    code = book()
    tools.cancel_reservation(code, "Maria")
    return {"code": code}


def fill_olive_fork():
    book("Big Party", OLIVE, "19:00", 20)
    return {}


# First-person "I confirm / we confirmed / we booked" forms (incl. the Russian-like "подтверждавам" Gemma produced).
EARLY_CONFIRMATION = re.compile(r"(под|по)тв[еъ]р(жд|д)ава(м|ме)|потвърдихме|запотвърд|потвърдена|резервирахме|"
                                r"запазихме|резервирах|запазих")

NEGATIVE = ("can't", "cannot", "not able", "unable", "don't", "do not", "only help", "not possible", "sorry",
            "не мога", "нямам", "за съжаление")


@dataclass
class Case:
    id: str
    prompt: str
    check: Callable[[dict], bool]
    note: str
    history: tuple = ()                        # earlier (role, text) turns; may use {code} from setup
    setup: Callable[[], dict] | None = None  # seeds data; returned values fill {placeholders}
    now: datetime | None = None             # clock for this case (default: NOW)
    lead_in: tuple = ()                        # earlier user turns that really run through the model first,
                                               # so the history holds real tool calls and results
    then: tuple = ()                           # fixed (role, text) turns after the lead-in, before the prompt


CASES = [
    # --- search ---
    Case("search-bg", "Търся италиански ресторант в центъра на София.",
         lambda r: any(a.get("cuisine") == "Italian" and a.get("city") == "Sofia" for a in calls(r, "search_restaurants"))
         and cyrillic(r["reply"]) and MARBLE[1] in r["reply"],
         "Bulgarian search uses English filters, replies in Bulgarian with the top result"),
    Case("names-not-translated-bg", "Търся мексикански ресторант във Варна.",
         lambda r: "Old Mill" in r["reply"] and "Стар" not in r["reply"],
         "keeps restaurant names as they are (Qwen said 'Стар Мил' for 'Old Mill' in testing)",
         now=datetime(2026, 11, 22, 1, 20)),  # the clock of the live test where it was seen
    Case("search-open-late", "Which Japanese restaurants in Berlin are still open at 23:30?",
         lambda r: any(a.get("open_at") == "23:30" for a in calls(r, "search_restaurants")),
         "passes the time as open_at"),
    Case("search-by-name", "Where exactly is Marble Hearth and when is it open?",
         lambda r: any("marble" in (a.get("name") or "").lower() for a in calls(r, "search_restaurants"))
         and "17:00" in r["reply"] and "23:00" in r["reply"],
         "finds a restaurant by name and gives real opening hours"),
    Case("unknown-city", "Can you recommend a restaurant in Paris?",
         lambda r: not reservations(r) and mentions(r["reply"], "sofia", "plovdiv", "varna", "london", "berlin"),
         "does not invent a Paris restaurant; names the real cities"),
    Case("german", "Ich suche ein französisches Restaurant in Berlin.",
         lambda r: called(r, "search_restaurants") and detect_text_language(r["reply"]) == "de",
         "replies in German"),

    # --- booking flow ---
    Case("relative-date", "Is there a free table for 2 at Marble Hearth tomorrow at 19:30?",
         lambda r: any(a.get("date") == TOMORROW and a.get("time") == "19:30" and a.get("guests") == 2
                       for a in calls(r, "check_availability")) and not reservations(r),
         "'tomorrow' becomes 2026-11-21; only checks, does not book"),
    Case("no-booking-without-confirmation", "Book a table for 2 at Marble Hearth tomorrow at 19:30 under the name Maria.",
         lambda r: not reservations(r) and called(r, "check_availability") and "?" in r["reply"],
         "checks availability and asks to confirm instead of booking straight away"),
    Case("no-early-confirmation-bg", "Първият звучи добре. Искам маса за двама в 19:30 на името на Мария Петрова.",
         lambda r: not reservations(r) and "?" in r["reply"] and not EARLY_CONFIRMATION.search(r["reply"].lower()),
         "before booking, says the table is available and asks; never claims it is already confirmed (seen in testing)",
         lead_in=("Здравейте! Търся италиански ресторант в центъра на София за утре вечер.",),
         now=datetime(2026, 11, 22, 0, 40)),  # the late-night clock of the live test where it was seen
    Case("booking-uses-restaurant-in-context-bg",
         "Искам да направя резервация за след 4 дни, за 5 човека от 20:30. Да е на името на Тихомир Несторов.",
         lambda r: not called(r, "book_table")
         and any(a.get("restaurant_id") == "REST-0389" for a in calls(r, "check_availability"))
         and "Rustic Cellar" in r["reply"],
         "books at the restaurant the guest was just asking about instead of asking which one (seen in testing)",
         setup=seed_booking,
         lead_in=("Искам да преместя резервация {code} в 19:30, на името на Мария е.",
                  "Търся гръцки ресторант в София.", "какво е менюто в Rustic Cellar")),
    Case("book-after-confirmation", "Yes, please book it.",
         lambda r: one_booking(r, restaurant=MARBLE[1], date=TOMORROW, time="19:30", guests=2, customer_name="Maria")
         and reservations(r)[0]["code"] in r["reply"],
         "books after the guest confirms and gives the code exactly as stored",
         lead_in=("Book a table for 2 at Marble Hearth tomorrow at 19:30 under the name Maria.",)),
    Case("time-as-digits", "Маса за двама в Marble Hearth утре в 19:30, на името на Мария.",
         lambda r: "19:30" in r["reply"] and not mentions(r["reply"], "девет", "седем и половина", "половин"),
         "reads the time back as 19:30, not in words (a real bug found in testing)"),
    Case("full-slot-alternatives", "Do you have a table for 4 at Olive Fork tomorrow at 19:00?",
         lambda r: not last_result(r, "check_availability").get("available", True)
         and any(t in r["reply"] for t in last_result(r, "check_availability").get("alternatives", [])),
         "slot is full; offers alternative times from the tool",
         setup=fill_olive_fork),
    Case("past-date", "Book a table for 2 at Marble Hearth yesterday at 19:30 for Maria.",
         lambda r: not reservations(r) and mentions(r["reply"], "past", "passed", "yesterday", "future"),
         "refuses a time in the past"),
    Case("large-group", "We are 30 people. Table at Marble Hearth tomorrow at 19:00, name Ivan.",
         lambda r: not reservations(r) and not r["requests"] and "?" in r["reply"]
         and mentions(r["reply"], "phone", "email", "e-mail", "contact", "call you back"),
         "groups over 20 go to a human: asks how a colleague can reach them instead of promising a call"),
    Case("no-false-handoff-bg", "Над 20 души, искам да прехвърлите запитването към екипа ви.",
         lambda r: not r["requests"] and mentions(r["reply"], "телефон", "имейл", "email", "e-mail", "контакт"),
         "does not claim to have passed the request on before it has a way to reach the guest (seen in testing)",
         lead_in=("Имам въпрос относно организиране на детско парти в ресторант в София.",)),
    Case("event-callback-bg", "Да, моля.",
         lambda r: len(r["requests"]) == 1 and "0888" in r["requests"][0]["contact"]
         and "иван" in r["requests"][0]["customer_name"].lower() and mentions(r["reply"], "персонал", "колег", "екип", "свърж"),
         "saves a callback request with the name and phone, so a colleague really can call back",
         lead_in=("Имам въпрос относно организиране на детско парти в ресторант в София.",
                  "Ще са около 25 деца, в събота следобед. Казвам се Иван Петров, телефонът ми е 0888 123 456.")),
    Case("person-request-mid-booking-bg", "Не искам повече да говоря с робот. Искам да говоря с човек.",
         lambda r: not r["requests"] and not reservations(r)
         and mentions(r["reply"], "телефон", "имейл", "email", "e-mail", "контакт")
         and mentions(r["reply"], "персонал", "ресторант", "колег", "обад"),
         "a request for a person in the middle of a booking: the callback steps start (how can the staff reach "
         "you?) instead of the booking going on (by the agent, or by the wants-a-person gate)",
         lead_in=("Търся маса за двама утре вечер в София, италианска кухня.",)),
    Case("event-no-invented-team", "Hi! I'd like to book the whole restaurant at Marble Hearth for a company party "
                                   "of 45 people next Friday.",
         lambda r: not reservations(r) and not r["requests"] and no_invented_team(r)
         and mentions(r["reply"], "phone", "email", "e-mail", "contact", "reach"),
         "asks how the staff can reach the guest, without inventing an 'events team' (seen in a live test)"),
    Case("event-callback-staff-en", "Yes, that's right.",
         lambda r: len(r["requests"]) == 1 and "765" in r["requests"][0]["contact"] and no_invented_team(r)
         and mentions(r["reply"], "staff", "colleague", "restaurant"),
         "saves the callback and says the restaurant staff will call, not an 'events team' (seen in a live test)",
         lead_in=("Hi! I'd like to book the whole restaurant at Marble Hearth for a company party of 45 people "
                  "next Friday.",
                  "My name is Daniel Ivanov, phone +359 88 765 4321. It's a Christmas party with dinner and drinks.")),

    # --- menu and allergens ---
    Case("menu-in-guest-language-bg", "Какво има в менюто на Olive Fork?",
         lambda r: any(a.get("language") == "bg" for a in calls(r, "get_menu"))
         and sum(n.lower() in r["reply"].lower() for n in menu_names(OLIVE[0], "bg")) >= 2  # "телешко варено" mid-sentence
         and not any(n.lower() in r["reply"].lower() for n in menu_names(OLIVE[0], "en") if n not in menu_names(OLIVE[0], "bg")),
         "dish names in Bulgarian from the data, not English (asked for after a live test)"),
    Case("cheapest-by-price", "Which is the cheapest Italian restaurant in Sofia?",
         lambda r: cheapest(cuisine="Italian", city="Sofia") in r["reply"]
         and "$" not in r["reply"] and mentions(r["reply"], "€", "eur"),
         "names the cheapest by price per person, in euros, never '$' (was 'price class $')"),
    Case("allergen-gluten", "I'm allergic to gluten. What can I eat at Marble Hearth?",
         lambda r: any("gluten" in (a.get("avoid_allergens") or []) for a in calls(r, "get_menu"))
         and not any(d in r["reply"] for d in unsafe_dishes(MARBLE[0], "gluten"))
         and mentions(r["reply"], "staff", "waiter", "server", "tell", "inform"),
         "names only gluten-free dishes and reminds the guest to tell the staff"),
    Case("no-invented-dish", "Do you serve pizza at Grand Courtyard?",
         lambda r: called(r, "get_menu") and mentions(r["reply"], "no", "not", "don't", "doesn't"),
         "checks the real menu (Japanese, no pizza) instead of guessing"),

    # --- changes and cancellations ---
    Case("allergy-note", "Please add a note to my reservation {code}, name Maria: my friend is allergic to gluten.",
         lambda r: "gluten" in (reservations(r)[0]["notes"] or "").lower(),
         "adds the allergy to the reservation (the promise that used to have no tool)",
         setup=seed_booking),
    Case("move-reservation", "Could you move my reservation {code} to 20:30? It's under Maria.",
         lambda r: one_booking(r, time="20:30"),
         "moves the booking to a new time",
         setup=seed_booking),
    Case("add-name", "My reservation {code} is under Maria, but my colleague Peter Ivanov will come instead. "
                     "Can you put it under his name?",
         lambda r: one_booking(r, customer_name="Maria")
         and any("peter" in n.lower() for n in reservations(r)[0]["extra_names"]),
         "adds the new name and keeps the original one (names are added, never replaced)",
         setup=seed_booking),
    Case("cancel-needs-name", "Cancel reservation {code}.",
         lambda r: reservations(r)[0]["status"] == "confirmed" and mentions(r["reply"], "name")
         and "maria" not in r["reply"].lower(),
         "does not cancel without the booking name and does not reveal it (Qwen said 'booked under Maria')",
         setup=seed_booking),
    Case("cancel-after-confirmation", "Yes.",
         lambda r: reservations(r)[0]["status"] == "cancelled",
         "cancels after the guest confirms",
         setup=seed_booking,
         history=(("user", "Please cancel reservation {code}, the name is Maria."),
                  ("assistant", "I found reservation {code} for Maria at Marble Hearth on 2026-11-21 at 19:00. "
                                "Shall I cancel it?"))),

    Case("rebook-cancelled-bg", "Анулирах резервация {code} по грешка, на името на Мария е. Може ли да я възстановите?",
         lambda r: not called(r, "book_table")
         and any(booking_key_of(a) == (MARBLE[0], TOMORROW, "19:00", 2) for a in calls(r, "check_availability"))
         and "?" in r["reply"],
         "offers to book the same table again instead of only passing the guest to a colleague (seen in testing)",
         setup=seed_cancelled_booking),

    # --- scope ---
    Case("out-of-scope", "Can you also order me a taxi to the restaurant?",
         lambda r: not reservations(r) and mentions(r["reply"], *NEGATIVE),
         "says it cannot order a taxi instead of pretending"),
]


def run_case(case, model):
    shutil.copyfile(TEMPLATE_DB, config.DB_PATH)
    clock = case.now or NOW
    tools.now = lambda: clock
    context = case.setup() if case.setup else {}
    history = [{"role": role, "content": text.format(**context)} for role, text in case.history]
    tool_calls, lead_replies = [], []
    try:
        for text in case.lead_in:
            history.append({"role": "user", "content": text.format(**context)})
            lead_replies.append(agent.run_llm_turn(history, [], model=model).content or "")
            history.append({"role": "assistant", "content": lead_replies[-1]})
    except Exception as e:
        return {"case": case.id, "outcome": "error", "reply": "", "tools": [], "reservations": [], "requests": [], "seconds": 0,
                "error": f"lead-in failed: {e}"}
    history += [{"role": role, "content": text.format(**context)} for role, text in case.then]
    history.append({"role": "user", "content": case.prompt.format(**context)})
    started = time.time()
    try:
        msg = agent.run_llm_turn(history, tool_calls, model=model)
        reply, error = (msg.content or ""), None
    except Exception as e:
        reply, error = "", str(e)
    run = {"reply": reply, "lead_replies": lead_replies, "tools": tool_calls, "reservations": tools.list_reservations(), "requests": tools.list_requests(),
           "seconds": time.time() - started, "error": error}
    try:
        passed = error is None and bool(case.check(run))
    except Exception as e:
        passed, run["error"] = False, f"check failed: {e}"
    outcome = "pass" if passed else ("error" if error and not error.startswith("check failed") else "fail")
    return {"case": case.id, "outcome": outcome, **run}


def load_cases(path):
    """Cases from another file (a set kept outside the repo): it gets this module, to build Case objects and checks."""
    spec = importlib.util.spec_from_file_location("external_cases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.cases(sys.modules[__name__]))


def main():
    # Replies in Bulgarian or German must print on a Windows console or a redirected log (cp1252 by default).
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=[m for m, _ in config.ALLOWED_MODELS])
    parser.add_argument("--cases", nargs="*", help="run only these case ids")
    parser.add_argument("--repeat", type=int, default=1, help="runs per case (LLM output varies between runs)")
    parser.add_argument("--output", default="latest", help="name of the summary file in the results directory")
    parser.add_argument("--case-file", help="a Python file whose cases(runner) returns the cases to run instead of CASES")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="where the JSON and the Markdown summary go")
    args = parser.parse_args()
    cases = load_cases(args.case_file) if args.case_file else CASES
    cases = [c for c in cases if not args.cases or c.id in args.cases]
    labels = dict(config.ALLOWED_MODELS)

    data.build_database(TEMPLATE_DB)
    results = {}
    for model in args.models:
        print(f"\n=== {model}: loading in LM Studio ({len(cases)} cases x {args.repeat}) ...", flush=True)
        switched = lmstudio.switch_model(model)
        if switched["status"] != "success":
            print(f"    could not load: {switched['message']}")
            continue
        results[model] = []
        for case in cases:
            for _ in range(args.repeat):
                r = run_case(case, model)
                if r["outcome"] == "error":     # one retry after an LM Studio error, reloading the model
                    lmstudio.switch_model(model)
                    r = run_case(case, model)
                results[model].append(r)
                print(f"  {r['outcome'].upper():<5} {case.id:<34} {r['seconds']:5.1f}s  "
                      f"tools={[c['name'] for c in r['tools']]}", flush=True)
                if r["outcome"] != "pass":
                    print(f"        reply: {r['reply'][:300]!r}  error: {r['error']}", flush=True)
    lmstudio.switch_model(config.MODEL_NAME)  # leave LM Studio with the default model loaded

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    with open(os.path.join(results_dir, f"{'results' if args.output == 'latest' else args.output}-{stamp}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    lines = [f"# Eval results ({stamp})", "",
             f"{len(cases)} cases, {args.repeat} run(s) each, fresh demo data per run, clock frozen (Fri 2026-11-20 "
             "10:00 unless a case sets its own). Graded on reservations, tool calls and the reply. "
             "Generated by `python evals/run_evals.py`.", "",
             "| Model | Passed | Infra errors | Median time per turn |", "|---|---|---|---|"]
    for model, rows in results.items():
        passed = sum(r["outcome"] == "pass" for r in rows)
        errors = sum(r["outcome"] == "error" for r in rows)
        lines.append(f"| {labels.get(model, model)} | **{passed}/{len(rows)}** | {errors} | "
                     f"{statistics.median(r['seconds'] for r in rows):.1f} s |")
    lines += ["", "✅ pass · ❌ wrong behaviour · ⚠️ LM Studio / infrastructure error (not counted as a model failure)",
              "", "| Case | What is checked | " + " | ".join(labels.get(m, m) for m in results) + " |",
              "|---|---|" + "---|" * len(results)]
    for case in cases:
        cells = ["".join({"pass": "✅", "fail": "❌", "error": "⚠️"}[r["outcome"]]
                         for r in results[model] if r["case"] == case.id) for model in results]
        lines.append(f"| `{case.id}` | {case.note} | " + " | ".join(cells) + " |")
    with open(os.path.join(results_dir, f"{args.output}.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
