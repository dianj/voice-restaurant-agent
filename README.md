# Multilingual Voice AI Restaurant Concierge

A local-first, multilingual voice AI agent capable of handling phone-style restaurant inquiries, checking menu ingredients and allergens, and booking tables in real time.

## Architecture
```text
[Browser Mic (HTTPS)] 
       │ (WebM Audio)
       ▼
[FastAPI Server] ──► [Faster-Whisper STT] ──► (Text + Detected Language)
                                                      │
                                                      ▼
[SQLite DB (1000 rows)] ◄── (SQL Queries) ◄── [Gemma 4-26B (LM Studio)]
       │                                              │
       └──────────────► (Tool Result) ────────────────┘
                                                      │ (Multilingual Response)
                                                      ▼
[Browser Speaker] ◄── (MP3 Audio) ◄────────── [Edge-TTS Neural Voice]
```

## Tools Definition
1. `search_restaurants(cuisine, location)` — Queries the 1,000-restaurant SQLite database by cuisine and location, returning ratings, seating capacity, and opening hours.
2. `get_menu_and_allergens(restaurant_id)` — Retrieves dishes, full ingredient lists, and allergen warnings for a given restaurant.
3. `check_and_book_table(restaurant_id, customer_name, date, time, guests)` — Validates party size against restaurant capacity and records confirmed reservations in SQLite.

## Sample Multilingual Conversation
* **User (Bulgarian):** *"Здравей, търся италиански ресторант в центъра на София."*
  * **Tool Call:** `search_restaurants(cuisine="Italian", location="Sofia Center")`
  * **Agent (Bulgarian):** *"Намерих Golden Table #1 в центъра на София. Отворен е от 10:00 до 22:00 и има рейтинг 4.7. Искате ли да проверим менюто или да запазим маса?"*
* **User (Switches to English):** *"Yes, does the Truffle Risotto contain dairy? Also book a table for 2 tomorrow at 19:30 under Alex."*
  * **Tool Calls:** `get_menu_and_allergens(restaurant_id="REST-0001")` & `check_and_book_table(restaurant_id="REST-0001", customer_name="Alex", date="2026-11-21", time="19:30", guests=2)`
  * **Agent (English):** *"Yes, the Truffle Risotto contains dairy. I have confirmed your table for 2 guests tomorrow at 19:30 under the name Alex."*

## Failure Handling Strategy
* **Unintelligible Speech / Empty Audio (STT Fallback):** If `faster-whisper` returns an empty or <2-character transcript, the backend bypasses the LLM and politely asks the user to repeat in the detected language.
* **Business Logic Constraints:** If a user requests a non-existent `restaurant_id` or a party size exceeding `seating_capacity`, the SQL tool returns a structured failure reason (`{"status": "failed", "reason": "..."}`) so Gemma 4 can apologize and suggest alternatives without hallucinating.
* **Database or LLM Outage:** All SQLite operations and LM Studio API calls are wrapped in `try-except` blocks, gracefully escalating the call to a human operator if a system error occurs.

## Quick Start
```bash
pip install -r requirements.txt
python setup_db.py
uvicorn agent_server:app --host 0.0.0.0 --port 8000
```
To expose publicly via HTTPS for live testing:
```bash
cloudflared tunnel --url http://localhost:8000
```
