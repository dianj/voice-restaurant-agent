# Architecture and Developer Guide: Multilingual Voice Restaurant Concierge

This guide explains how the project works: request flow, data model, tools and booking rules, prompt, voice, frontend, testing and known limitations. The [README](../README.md) is the short overview. The sequence diagram of one chat turn is in [`voice-restaurant-agent_diagram.svg`](../voice-restaurant-agent_diagram.svg).

---

## 1. Quick start

1. **LM Studio:** start the local server on **port 1235** (not the default 1234) and load `gemma-4-26b-a4b-it`.
2. **Python 3.12:**
   ```powershell
   pip install -r requirements.txt
   uvicorn server:app --host 127.0.0.1 --port 8001
   ```
3. Open `http://localhost:8001`. The microphone works on `localhost` or over HTTPS only.

The demo database is created on first start. `python setup_db.py` rebuilds it and **deletes all reservations**; stop the server first.

| Variable | Default |
|---|---|
| `LLM_BASE_URL` | `http://localhost:1235/v1` |
| `LLM_MODEL` | `gemma-4-26b-a4b-it` (default; the UI can switch to `qwen3.8-27b`) |
| `LLM_CONTEXT_LENGTH` | `16384` (used when the app loads a model) |
| `WHISPER_MODEL` | `large-v3-turbo` |
| `RESTAURANT_DB_PATH` | `restaurant_agent.db` in the project root |

---

## 2. Code layout

| Path | Purpose |
|---|---|
| `server.py` | FastAPI endpoints and in-memory sessions. Thin layer over the `restaurant` package. |
| `restaurant/config.py` | Environment settings, business rules (sitting length, last seating, limits), voices |
| `restaurant/data.py` | Schema, deterministic demo data (seed 42), `ensure_database()` |
| `restaurant/tools.py` | The eight tools and the booking rules |
| `restaurant/tool_schemas.py` | Tool schemas sent to the model |
| `restaurant/registry.py` | `execute_tool_call`, which never raises |
| `restaurant/prompts.py` | System prompt with today's date |
| `restaurant/agent.py` | Tool-calling loop, booking confirmation gate, history trimming |
| `restaurant/lmstudio.py` | LM Studio model list, load, unload and switch |
| `restaurant/voice.py` | faster-whisper STT with a restricted language pick, text language detection, edge-tts |
| `index.html` | Single page, vanilla JS: conversation pane and staff view |
| `setup_db.py` | Rebuilds the database and exports `restaurants_1000.csv` |
| `tests/` | Deterministic tests with a scripted fake LM Studio (`fake_lmstudio.py`) |
| `evals/run_evals.py` | Real-model evaluation; writes `evals/results/latest.md` |

Dependencies go one way: `server` → `agent` → `registry` → `tools` → `data` / `config`. `voice` is used only by `server`.

---

## 3. One chat turn

1. `POST /api/chat` receives `message` (text) or `audio` (a recording).
2. Audio is limited to 15 MB and, after decoding, to `MAX_AUDIO_SECONDS = 60` (longer → HTTP 413 with a friendly message, Whisper is not run), written to a temp file and transcribed in a worker thread. An empty transcript gets a "please repeat" reply in the detected language without calling the LLM.
3. Under the session's lock, the turn works on a **copy** of the history plus the new user message.
4. `run_llm_turn` sends the system prompt, history and 8 tool schemas. Tool calls are executed and their JSON results sent back, for up to `MAX_TOOL_ROUNDS = 6` rounds. Hitting the limit gives a "technical problem, please try again" reply (it promises nothing: no one could take a callback then).
5. On success the copy (with the reply) replaces the session history, trimmed to the last 12 user turns. Trimming cuts only before a user message, so tool-call pairs stay complete. On failure the history is left as it was.
6. The response contains `reply`, `reply_language`, `spoken_language`, `user_text` and `tool_calls` (name, arguments, status). Exceptions are logged; the guest sees a fixed message in their language and `system_error: true`.
7. The browser refreshes the staff view (`GET /api/reservations`, with the staff token) and, with voice replies on, fetches `POST /api/tts`. The reply's voice language is the reply's own when it has enough words to tell, otherwise the guest's (spoken, or of their message; kept in the session).

| Endpoint | Purpose |
|---|---|
| `GET /` | The page |
| `POST /api/chat` | One turn (text or audio) |
| `POST /api/tts` | Reply text → MP3 (`language`, `gender`); edge-tts is free, so the text stays open, within the chat rate limit |
| `POST /api/reset` | Clear this session's history; without a session nothing is created (204) |
| `GET /api/reservations` | Staff view: latest 50 reservations and latest 50 callback requests; needs `X-Staff-Token` (401 otherwise; 10 wrong tokens per minute per address, then 429) |
| `GET /api/health` | Is LM Studio reachable, the active model loaded, a switch running (not the LM Studio address) |
| `GET /api/models` · `POST /api/model` | List the models; switch the loaded model |

**Sessions:** a cookie (`restaurant_session`) maps to `{history, lock, language}` in an `OrderedDict` capped at 200 sessions (least recently used are dropped). Only a chat message makes a new session (a reset without a cookie does not, so strangers cannot push a guest out mid-booking). They are lost on restart.

**Request limits** (`restaurant/ratelimit.py`): a token bucket per client address, 60 per minute for chat and speech together (`CHAT_PER_MINUTE`), 10 wrong staff tokens per minute (`STAFF_PER_MINUTE`); over the limit → 429 with `Retry-After`. Behind a reverse proxy the address must come from the proxy (`uvicorn --proxy-headers`).

**Staff token:** `STAFF_TOKEN`, or a random one made at start and printed once in the console. The staff view beside the chat asks for it and keeps it for the browser tab (sessionStorage).

---

## 4. Data model

* `restaurants`: `restaurant_id` (`REST-0001`…), unique `name`, `city`, `area`, `cuisine`, `price_range` (internal: scales menu prices; never shown to the agent), `price_per_person` (typical EUR per person = two courses at the menu's median price, rounded to 5), `rating` (NULL for about 4% = "not rated yet"), `seating_capacity`, `opening_hours` (`"12:00-00:00"`; a closing time at or before the opening time means after midnight), plus analytics columns with deliberate gaps and outliers (`review_count`, `average_meal_price`, `monthly_revenue`, `operating_costs`, `opening_year`). The agent never reads the analytics columns.
* `menu_items`: dishes from a per-cuisine template (4–6 per restaurant; 7–10 for Bulgarian, including телешко варено, пилешка супа, шкембе чорба, мусака and сарми); `dish_name` (English) and `names` (JSON with the name in en, bg, de, es, fr, it); `allergens` uses the 14 EU-declared allergen names (stored in English; `data.ALLERGEN_NAMES` has them in all six languages, so `get_menu` returns "целина", not the model's own "селера"); `vegetarian`, `vegan`, `price` (scaled by price range).
* `requests`: callback requests (`kind`, `customer_name`, `contact`, `details`, optional `date`, `guests`, `city`, `status`, `created_at`); created in place on start if missing.
* `ensure_database()` rebuilds the database when a required column is missing, which deletes reservations (demo data). The later reservation columns `extra_names` and `history` are added in place instead, so existing bookings survive.
* `reservations`: `code` (unique, e.g. `R34LST`), `restaurant_id`, `customer_name` (the name it was booked under; never replaced), `extra_names` (JSON list of added names), `phone`, `date`, `time`, `guests`, `notes`, `history` (JSON list of changes with time), `status` (`confirmed` | `cancelled`), timestamps. Index on `(restaurant_id, date, status)`.

Locations: Sofia (Center, Lozenets, Vitosha Blvd), Plovdiv (Old Town, Kapana), Varna (Sea Garden, Center), London (Soho), Berlin (Mitte). Cuisines: Italian, Bulgarian, Japanese, Mexican, French, Indian, Mediterranean, Steakhouse.

`ensure_database()` rebuilds the database when it is missing or still has the prototype schema (no `code` column).

---

## 5. Tools and booking rules

| Tool | What it does |
|---|---|
| `search_restaurants` | Filters: `name` (partial), `cuisine`, `city`, `area`, `min_rating`, `max_price_per_person`, `open_at` (HH:MM, handles past-midnight hours). `sort_by`: rating (unrated last) or price. Each result has `price_per_person` ("about 20 EUR"); up to 10 results. Unknown values return the allowed list. |
| `get_menu` | Dishes in the requested `language` (stored translations, English fallback), prices as "6.00 EUR", with allergens; `avoid_allergens` (everyday words such as "dairy" or "wheat" are mapped) marks `contains_avoided` and lists `safe_dishes` with a safety note. Vegetarian/vegan filters. |
| `check_availability` | Validates the request and checks seats; does not book. When full, returns up to 3 nearest free times. |
| `book_table` | Same checks inside `BEGIN IMMEDIATE`, then inserts. Needs a name (2–80 characters). Returns the code. |
| `find_reservation` | Lookup by code, with the restaurant ID so a cancelled booking can be checked and booked again under a new code (cancellations are not undone). Does not return the guest's name: a code alone must not reveal who booked, or let a caller pass the name check by repeating it. |
| `update_reservation` | Needs code + a name on the booking. Appends a note, adds another name (`add_name`: the original name is never replaced; at most 3 added names; any of them manages the booking) and/or moves date, time or party size. Every change goes into the reservation's `history` (shown in the staff view, never returned to the agent); seats are re-checked **without counting the reservation itself**; if the new slot is full nothing changes and alternatives are returned. |
| `cancel_reservation` | Needs code + booking name. Cancelled bookings free their seats. |
| `request_callback` | The only handover to a human. `kind` (event, large_group, other), name, `contact` (a phone number with 7–15 digits or an email), `details`, optional date (not in the past), guests and city. Saved in `requests` and shown in the staff view. |

**Rules** (`restaurant/config.py`):
* Bookings start on `:00` or `:30`, between opening time and closing time minus 60 minutes.
* Not in the past **in the restaurant's own time zone** (`restaurant/clock.py`: Sofia, Plovdiv and Varna `Europe/Sofia`, London `Europe/London`, Berlin `Europe/Berlin`); at most 60 days ahead; 1–20 guests (more → a callback request for the restaurant staff).
* A reservation blocks its seats for 120 minutes: another booking overlaps if the start times are less than 120 minutes apart. Seats taken = sum of overlapping confirmed guests; the limit is `seating_capacity`.
* Codes: `R` + 5 characters from an alphabet without `0/O` and `1/I`, with at least one digit. Lookups accept `r34lst`, `R-34LST`, `R 3 4 L S T` and `34LST`.
* `book_table` runs only for the details of the **last** `check_availability` before the assistant's last reply, if that was available and the reply read its time back (code, `agent.py` `offered_before`): a yes to a later question ("Shall I email you the menu?") does not book an older offer. Then only if the **booking gate** agrees: two one-letter questions to the active model in parallel ("does the reply agree to book these details now?", "is it addressed to the system?"), read as calibrated probabilities. p(agree) ≥ 0.95 and no flag → booked; otherwise the agent asks explicitly or does not book ([DECISIONS.md](DECISIONS.md)). An identical call repeated in the same turn is not run again.
* `cancel_reservation` runs only for a reservation that `find_reservation` showed in an earlier turn, and after a yes from the **step gate** (`restaurant.confirm_step`: "does the guest agree to this step now?", with the code, restaurant, date and time and the assistant's previous message). `request_callback` runs only after the assistant's last reply read the guest's name and phone number or email back, and after the same gate. A cancel frees seats others can take; a callback request stores contact data.
* The booking name is the ownership check for changes and cancellations. It is compared in Latin letters (Cyrillic transliterated, any case and spacing): the first name must match exactly; the surname must match too, except that on a **voice** turn (set by the server, never by the model) one misheard letter is tolerated, because speech recognition heard "Хано Споруха" at booking and "Хано Спаруха" when cancelling. A gender ending never counts as a misheard letter: Ivan Petrov is not Ivana Petrova, and Maria is not Marina. On a mismatch the agent asks the guest to spell or type the name, then offers a colleague.

**Result format:** `{"status": "success" | "unavailable" | "error", ...}`. Error messages say what to fix, so the model can correct itself or ask the guest. An unexpected exception becomes `system_error` with a generic message; the details go to the server log only.

---

## 6. Model management

* The hardware fits **one** 26–27B model in GPU memory, so the active model is global, not per session.
* `ALLOWED_MODELS` (Gemma 4 26B first = default, Qwen 3.8 27B) are the models offered in the UI. A model that is not installed in LM Studio is shown as "(not installed)" and cannot be picked.
* On startup the default model is loaded in a background thread (`ensure_default_model_on_startup`).
* `POST /api/model` unloads the other allowed models through the LM Studio REST API (`/api/v1/models/unload`), then loads the target (`/api/v1/models/load`, context `LLM_CONTEXT_LENGTH`). If that fails, the previous model is restored.
* While a switch runs, chat answers "one moment" in the guest's language without calling the LLM. A switch is refused (HTTP 409) while any turn is waiting for the LLM, so the model that is answering is never unloaded; the browser also disables the selector during its own turns.
* LM Studio is shared: the website concierge (also on port 1235) defaults to Gemma too. Switching here affects every app that uses LM Studio.

## 7. System prompt (`build_system_prompt` in `restaurant/prompts.py`)

* **Today's date, weekday and time per time zone** come from `restaurant/clock.py` (UTC inside, frozen in tests and evals; a test fails on a naive `datetime.now()` anywhere else in the package), so "tomorrow" and "on Friday" resolve in the restaurant's city: after midnight in Sofia it is still the evening before in London. Stored timestamps are UTC with an offset; the staff view shows them in the browser's local time.
* **Language:** reply in the language of the guest's latest message; for voice turns the detected language is added.
* **Style:** read aloud, so 1–3 short sentences, no Markdown or IDs. **Times as digits exactly as in the tool result**: an earlier "say times naturally" rule made Gemma read 19:30 back as "half past nine".
* **Facts** only from tool results; pass `open_at` when the guest says when they want to eat.
* **Names:** restaurant names exactly as in the tool result, never translated (Qwen once turned "Old Mill" into "Стар Мил"); dish names from `get_menu` in the guest's language. **Prices:** per person in euros, never `$` signs.
* **Booking:** check, read the details back, book only after a yes, give the code exactly as written. Until `book_table` succeeds, never say booked, reserved or confirmed.
* **Enforced in code** (`guard` in `agent.py`): `book_table` runs only for the offer the assistant last read back (see §5); otherwise the model gets "Not booked: ask the guest first". Added after Gemma booked without asking in 2 of 2 eval runs once the prompt had grown. Cancels and callback requests are two steps in the same way.
* **Allergies:** filtered menu, remind to tell the staff, offer to add a note.
* **Never promise an action no tool can do.**
* **Human handover:** the agent cannot transfer calls. It collects name, phone or email and the request, reads them back and calls `request_callback` after a yes; until that succeeds it never says it passed anything on. Added after the agent told a guest it would forward a party request to an events team that did not exist.
* **Errors:** fix or ask on `error`; offer a callback on `system_error`.

---

## 8. Voice

* **STT:** faster-whisper `large-v3-turbo` on the CPU (int8), lazy-loaded on the first voice request (the first one is slow). Language = the most probable of EN, BG, DE, ES, FR, IT (short Bulgarian clips are otherwise labelled RU or MK). A per-language vocabulary prompt helps with booking terms (the Bulgarian one lists the dish names; "телешко варено" was heard as "телешко в арен"). VAD is on. The detection probabilities are logged.
* **Speech language selector** (next to the microphone): Auto-detect or a fixed language. A fixed language skips detection; added after Bulgarian turns were heard as Spanish and Italian and the agent answered in those languages.
* **Reply language for TTS:** Cyrillic → Bulgarian; otherwise a small stop-word score for EN/DE/ES/FR/IT.
* **TTS:** edge-tts, an **online Microsoft service** (the reply text leaves the machine). One male and one female voice per language. Reservation codes are spelled out for the voice only (`R34LST` → "R, 3, 4, L, S, T"). Three attempts with a 15 s timeout each.

---

## 9. Frontend (`index.html`)

* Conversation: text box (Enter sends, Shift+Enter adds a line), microphone button (click to start, click to send, 30 s maximum, microphone released after use, friendly message when access is blocked), example prompts, "New conversation", voice replies on/off and voice gender (saved in localStorage), "Read aloud" per reply, elapsed-seconds indicator.
* Each reply shows its tool calls as chips coloured by status; a click shows the arguments.
* While the agent works, a bubble with animated dots and a seconds counter ("The agent is thinking… 4 s") sits in the conversation where the reply will appear; it is a `role="status"` element, and the dots stop moving with `prefers-reduced-motion`.
* Staff view: a second table, "Callback requests", lists name, contact, the request with its kind, size, date and city, and when it came in.
* Staff view: reservations with code, guest (with added names) and notes, a "changed" badge with the list of changes (time in the tooltip), restaurant and area, time, guests and status; new rows flash; cancelled rows are struck through.
* All model and user text is rendered with DOM nodes and `textContent`, never as HTML.
* Header: model selector (Gemma / Qwen) with the status dot inside it (green = ready, red = loading, not loaded or LM Studio not reachable). The status text goes to the tooltip and to screen readers, since colour alone is not accessible.
* Text size: A− / A+ from 12 to 24 px in 2 px steps, saved in localStorage (`fontSize`). One CSS variable (`--fs`) scales the messages, input box, tool details, example prompts and the staff view.

**localStorage keys:** `voiceReplies`, `voiceGender`, `speechLanguage`, `fontSize`.

---

## 10. Tests and evals

**Deterministic tests** (54: 22 tool, 17 API, 15 Playwright; about 11 s; no GPU or LM Studio):
```powershell
pip install -r requirements-dev.txt
python -m pytest --browser-channel msedge   # local run on the installed Edge
python -m pytest                            # needs: python -m playwright install chromium
```
* `tests/conftest.py` sets `RESTAURANT_DB_PATH` and `LLM_BASE_URL` before importing `server`, builds one template database, copies it for every test, freezes `tools.now` at Friday 2026-11-20 10:00 and runs the real app with uvicorn on a free port.
* A mutation check was done: making `seats_taken` always return 0 fails the capacity tests.
* CI (`.github/workflows/tests.yml`) runs everything on every push.

**LLM evals** (`python evals/run_evals.py [--repeat N] [--cases ...]`, LM Studio required):
* 40 conversations in English, Bulgarian, German, Spanish, French and Italian, each on fresh data with a frozen clock (a case can set its own clock, given as Sofia time), graded on reservations, tool calls and the reply. `--case-file` runs another set (e.g. one kept outside the repo) with the same runner.
* `lead_in` turns run through the model before the graded turn (and fixed `then` turns after them), so the history holds real tool calls and results. A synthetic text history did **not** reproduce the early-confirmation problem below; real lead-in turns and the late-night clock did.
* Problems found and fixed: no way to look a restaurant up by name; Gemma dropping the dash when reading back `R-34LST` (codes no longer have a dash); Gemma saying "Подтверждавам резервация" ("I confirm the reservation") before booking, seen near midnight (prompt rule; the eval failed 2/4 runs before and 0/4 after); Qwen translating restaurant names ("Стар Мил" for "Old Mill"; 1/4 runs before, 0/6 after, a rare failure); Gemma booking without asking after the prompt grew (2/2 runs; now blocked in code, see §7); an "events team" that does not exist in the handover wording (0/6 → 6/6 after the prompt and tool text said "the restaurant staff"); Qwen repeating an identical `book_table` call in one round, which booked two tables (1/2 runs; identical calls in a turn now run once, 4/4 after).
* `--models` (default: both allowed models) loads each model in turn and loads Gemma again at the end.
* Last full run (2026-11-23, 30 cases × 2): Gemma 4 26B **60/60**, median 8.5 s per turn; Qwen 3.8 27B **59/60**, median 34.5 s; the failure (a duplicate booking) is fixed and the case re-measured 4/4. An earlier Qwen failure (it read out the guest's name after looking up a code) was fixed by no longer returning the name from `find_reservation`. Results: `evals/results/latest.md`.

---

## 11. Gotchas

* **LM Studio port is 1235.** A red dot in the header means LM Studio is not reachable or the model is not loaded.
* **`av<19`:** faster-whisper 1.2.x calls `av.open(..., metadata_errors=...)`, which PyAV 19 removed.
* **Windows console and Cyrillic:** set `PYTHONIOENCODING=utf-8` when printing Bulgarian text from scripts.
* **`setup_db.py` while the server runs:** the server keeps working, but all reservations are gone.
* Evals switch the model in LM Studio; a running app server keeps showing its own active model until the next switch. Do not chat with the app while evals run.

---

## 12. Known limitations

* Staff authentication is one shared token (demo level); no user accounts or roles.
* Sessions and request limits are in memory and lost on restart; the 200-session cap has no time-based expiry.
* The booking name is a weak ownership check; a real system would confirm by phone or email.
* Seats are counted per restaurant, not per table.
* TTS is online.
* For a public demo the app would need HTTPS, real staff accounts and persistent storage.

## 13. Ideas for next steps

* Confirmation by SMS or email, and a phone-number check for changes.
* Table-level seating (2-, 4- and 6-seat tables) instead of one capacity number.
* Streaming replies and streaming TTS to cut the wait in voice mode.
* Evals for real recorded audio (Spanish, French and Italian have two text cases each now).
