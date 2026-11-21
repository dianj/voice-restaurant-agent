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
| `LLM_MODEL` | `gemma-4-26b-a4b-it` |
| `WHISPER_MODEL` | `large-v3-turbo` |
| `RESTAURANT_DB_PATH` | `restaurant_agent.db` in the project root |

---

## 2. Code layout

| Path | Purpose |
|---|---|
| `server.py` | FastAPI endpoints and in-memory sessions. Thin layer over the `restaurant` package. |
| `restaurant/config.py` | Environment settings, business rules (sitting length, last seating, limits), voices |
| `restaurant/data.py` | Schema, deterministic demo data (seed 42), `ensure_database()` |
| `restaurant/tools.py` | The seven tools and the booking rules |
| `restaurant/registry.py` | Tool schemas sent to the model; `execute_tool_call` never raises |
| `restaurant/agent.py` | System prompt with today's date, tool-calling loop, history trimming |
| `restaurant/voice.py` | faster-whisper STT with a restricted language pick, text language detection, edge-tts |
| `index.html` | Single page, vanilla JS: conversation pane and staff view |
| `setup_db.py` | Rebuilds the database and exports `restaurants_1000.csv` |
| `tests/` | Deterministic tests with a scripted fake LM Studio (`fake_lmstudio.py`) |
| `evals/run_evals.py` | Real-model evaluation; writes `evals/results/latest.md` |

Dependencies go one way: `server` → `agent` → `registry` → `tools` → `data` / `config`. `voice` is used only by `server`.

---

## 3. One chat turn

1. `POST /api/chat` receives `message` (text) or `audio` (a recording).
2. Audio is limited to 15 MB, written to a temp file and transcribed in a worker thread. An empty transcript gets a "please repeat" reply in the detected language without calling the LLM.
3. Under the session's lock, the turn works on a **copy** of the history plus the new user message.
4. `run_llm_turn` sends the system prompt, history and 7 tool schemas. Tool calls are executed and their JSON results sent back, for up to `MAX_TOOL_ROUNDS = 6` rounds. Hitting the limit gives a "transfer to a colleague" reply.
5. On success the copy (with the reply) replaces the session history, trimmed to the last 12 user turns. Trimming cuts only before a user message, so tool-call pairs stay complete. On failure the history is left as it was.
6. The response contains `reply`, `reply_language`, `spoken_language`, `user_text` and `tool_calls` (name, arguments, status). Exceptions are logged; the guest sees a fixed message in their language and `system_error: true`.
7. The browser refreshes the staff view (`GET /api/reservations`) and, with voice replies on, fetches `POST /api/tts`.

| Endpoint | Purpose |
|---|---|
| `GET /` | The page |
| `POST /api/chat` | One turn (text or audio) |
| `POST /api/tts` | Reply text → MP3 (`language`, `gender`) |
| `POST /api/reset` | Clear this session's history |
| `GET /api/reservations` | Staff view (latest 50) |
| `GET /api/health` | Is LM Studio reachable and the model loaded |

**Sessions:** a cookie (`restaurant_session`) maps to `{history, lock}` in an `OrderedDict` capped at 200 sessions (least recently used are dropped). They are lost on restart.

---

## 4. Data model

* `restaurants`: `restaurant_id` (`REST-0001`…), unique `name`, `city`, `area`, `cuisine`, `price_range`, `rating` (NULL for about 4% = "not rated yet"), `seating_capacity`, `opening_hours` (`"12:00-00:00"`; a closing time at or before the opening time means after midnight), plus analytics columns with deliberate gaps and outliers (`review_count`, `average_meal_price`, `monthly_revenue`, `operating_costs`, `opening_year`). The agent never reads the analytics columns.
* `menu_items`: 4–6 dishes per restaurant from a per-cuisine template; `allergens` uses the 14 EU-declared allergen names; `vegetarian`, `vegan`, `price` (scaled by price range).
* `reservations`: `code` (unique, e.g. `R34LST`), `restaurant_id`, `customer_name`, `phone`, `date`, `time`, `guests`, `notes`, `status` (`confirmed` | `cancelled`), timestamps. Index on `(restaurant_id, date, status)`.

Locations: Sofia (Center, Lozenets, Vitosha Blvd), Plovdiv (Old Town, Kapana), Varna (Sea Garden, Center), London (Soho), Berlin (Mitte). Cuisines: Italian, Bulgarian, Japanese, Mexican, French, Indian, Mediterranean, Steakhouse.

`ensure_database()` rebuilds the database when it is missing or still has the prototype schema (no `code` column).

---

## 5. Tools and booking rules

| Tool | What it does |
|---|---|
| `search_restaurants` | Filters: `name` (partial), `cuisine`, `city`, `area`, `min_rating`, `open_at` (HH:MM, handles past-midnight hours). Sorted by rating, unrated last; up to 10 results. Unknown values return the allowed list. |
| `get_menu` | Dishes with allergens; `avoid_allergens` (everyday words such as "dairy" or "wheat" are mapped) marks `contains_avoided` and lists `safe_dishes` with a safety note. Vegetarian/vegan filters. |
| `check_availability` | Validates the request and checks seats; does not book. When full, returns up to 3 nearest free times. |
| `book_table` | Same checks inside `BEGIN IMMEDIATE`, then inserts. Needs a name (2–80 characters). Returns the code. |
| `find_reservation` | Lookup by code. |
| `update_reservation` | Needs code + booking name. Appends a note and/or moves date, time or party size; seats are re-checked **without counting the reservation itself**; if the new slot is full nothing changes and alternatives are returned. |
| `cancel_reservation` | Needs code + booking name. Cancelled bookings free their seats. |

**Rules** (`restaurant/config.py`):
* Bookings start on `:00` or `:30`, between opening time and closing time minus 60 minutes.
* Not in the past; at most 60 days ahead; 1–20 guests (more → the events team, i.e. a human).
* A reservation blocks its seats for 120 minutes: another booking overlaps if the start times are less than 120 minutes apart. Seats taken = sum of overlapping confirmed guests; the limit is `seating_capacity`.
* Codes: `R` + 5 characters from an alphabet without `0/O` and `1/I`, with at least one digit. Lookups accept `r34lst`, `R-34LST`, `R 3 4 L S T` and `34LST`.
* The booking name is the ownership check for changes and cancellations (case-insensitive).

**Result format:** `{"status": "success" | "unavailable" | "error", ...}`. Error messages say what to fix, so the model can correct itself or ask the guest. An unexpected exception becomes `system_error` with a generic message; the details go to the server log only.

---

## 6. System prompt (`build_system_prompt` in `restaurant/agent.py`)

* **Today's date, weekday and time** come from `tools.now()` (frozen in tests and evals), so "tomorrow" and "on Friday" resolve correctly.
* **Language:** reply in the language of the guest's latest message; for voice turns the detected language is added.
* **Style:** read aloud, so 1–3 short sentences, no Markdown or IDs. **Times as digits exactly as in the tool result**: an earlier "say times naturally" rule made Gemma read 19:30 back as "half past nine".
* **Facts** only from tool results; pass `open_at` when the guest says when they want to eat.
* **Booking:** check, read the details back, book only after a yes, give the code exactly as written. Until `book_table` succeeds, never say booked, reserved or confirmed.
* **Allergies:** filtered menu, remind to tell the staff, offer to add a note.
* **Never promise an action no tool can do.**
* **Errors:** fix or ask on `error`; offer a human on `system_error`.

---

## 7. Voice

* **STT:** faster-whisper `large-v3-turbo` on the CPU (int8), lazy-loaded on the first voice request (the first one is slow). Language = the most probable of EN, BG, DE, ES, FR, IT (short Bulgarian clips are otherwise labelled RU or MK). A per-language vocabulary prompt helps with booking terms. VAD is on.
* **Reply language for TTS:** Cyrillic → Bulgarian; otherwise a small stop-word score for EN/DE/ES/FR/IT.
* **TTS:** edge-tts, an **online Microsoft service** (the reply text leaves the machine). One male and one female voice per language. Reservation codes are spelled out for the voice only (`R34LST` → "R, 3, 4, L, S, T"). Three attempts with a 15 s timeout each.

---

## 8. Frontend (`index.html`)

* Conversation: text box (Enter sends, Shift+Enter adds a line), microphone button (click to start, click to send, 30 s maximum, microphone released after use, friendly message when access is blocked), example prompts, "New conversation", voice replies on/off and voice gender (saved in localStorage), "Read aloud" per reply, elapsed-seconds indicator.
* Each reply shows its tool calls as chips coloured by status; a click shows the arguments.
* Staff view: reservations with code, guest and notes, restaurant and area, time, guests and status; new rows flash; cancelled rows are struck through.
* All model and user text is rendered with DOM nodes and `textContent`, never as HTML.
* Header dot: model reachable and loaded (`/api/health`).

---

## 9. Tests and evals

**Deterministic tests** (36: 15 tool, 11 API, 10 Playwright; about 9 s; no GPU or LM Studio):
```powershell
pip install -r requirements-dev.txt
python -m pytest --browser-channel msedge   # local run on the installed Edge
python -m pytest                            # needs: python -m playwright install chromium
```
* `tests/conftest.py` sets `RESTAURANT_DB_PATH` and `LLM_BASE_URL` before importing `server`, builds one template database, copies it for every test, freezes `tools.now` at Friday 2026-11-20 10:00 and runs the real app with uvicorn on a free port.
* A mutation check was done: making `seats_taken` always return 0 fails the capacity tests.
* CI (`.github/workflows/tests.yml`) runs everything on every push.

**LLM evals** (`python evals/run_evals.py [--repeat N] [--cases ...]`, LM Studio required):
* 20 conversations in English, Bulgarian and German, each on fresh data with a frozen clock (a case can set its own clock), graded on reservations, tool calls and the reply.
* `lead_in` turns run through the model before the graded turn, so the history holds real tool calls and results. A synthetic text history did **not** reproduce the early-confirmation problem below; real lead-in turns and the late-night clock did.
* Problems found and fixed: no way to look a restaurant up by name; Gemma dropping the dash when reading back `R-34LST` (codes no longer have a dash); Gemma saying "Подтверждавам резервация" ("I confirm the reservation") before booking, seen near midnight (prompt rule; the eval failed 2/4 runs before and 0/4 after).
* Last run: Gemma 4 26B **40/40** (two runs per case), median 7.1 s per turn. Results: `evals/results/latest.md`.

---

## 10. Gotchas

* **LM Studio port is 1235.** A red dot in the header means LM Studio is not reachable or the model is not loaded.
* **`av<19`:** faster-whisper 1.2.x calls `av.open(..., metadata_errors=...)`, which PyAV 19 removed.
* **Windows console and Cyrillic:** set `PYTHONIOENCODING=utf-8` when printing Bulgarian text from scripts.
* **`setup_db.py` while the server runs:** the server keeps working, but all reservations are gone.
* Run evals against a model that is already loaded; the script does not load or switch models.

---

## 11. Known limitations

* No authentication: anyone who can open the page sees the staff view.
* Sessions are in memory and lost on restart; the 200-session cap has no time-based expiry.
* No rate limits; the recording limit (30 s) is enforced in the browser only (the server limits size to 15 MB).
* The booking name is a weak ownership check; a real system would confirm by phone or email.
* Seats are counted per restaurant, not per table.
* TTS is online.
* For a public demo the app would need HTTPS, authentication for the staff view, rate limits and persistent storage.

## 12. Ideas for next steps

* Confirmation by SMS or email, and a phone-number check for changes.
* Table-level seating (2-, 4- and 6-seat tables) instead of one capacity number.
* Streaming replies and streaming TTS to cut the wait in voice mode.
* Evals for Spanish, French and Italian, and for real recorded audio.
