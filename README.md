# Multilingual Voice Restaurant Concierge

A voice and text AI concierge for a restaurant group. Guests speak or type in English, Bulgarian, German, Spanish, French or Italian. The agent finds restaurants, answers menu and allergen questions, and books or cancels tables. A staff view shows reservations as they are made.

The LLM runs locally in LM Studio and works only through validated tools: it cannot invent a free table or a dish, because availability, menus and bookings come from SQLite through rules written in Python.

![A booking in Bulgarian, then an allergen question in English, with the reservation in the staff view](docs/screenshots/02-allergens-en.png)

<sub>Screenshot from a real session with Gemma 4 26B in LM Studio.</sub>

## What it does

| Area | Details |
|---|---|
| **Input** | Text, or voice through the browser microphone → local faster-whisper (`large-v3-turbo`). Language detection is limited to the six supported languages, so short Bulgarian clips are not mislabelled as Russian. |
| **Search** | By cuisine, city, area, minimum rating and "open at" time (handles places open past midnight). 1,000 restaurants, sorted by rating, unrated last. |
| **Menu & allergens** | Every restaurant has a menu with the 14 EU-declared allergens. Everyday words ("dairy", "wheat", "shellfish") map to them. The agent names safe dishes and reminds the guest to tell the staff. |
| **Booking rules** | Opening hours and last seating, no past dates, at most 60 days ahead, 1–20 guests (bigger groups go to a human), 2-hour sittings that block seats, nearest free times when full. Write lock (`BEGIN IMMEDIATE`) so parallel bookings cannot overbook. Codes like `R-RM952` avoid letters that sound alike. |
| **Conversation** | Checks availability, reads the details back and books only after the guest confirms. Cancelling needs the code and the booking name. Today's date is in the prompt, so "tomorrow" or "on Friday" resolve correctly. |
| **Voice output** | edge-tts with a male or female voice per language (online Microsoft service). |
| **Robustness** | Per-browser sessions, bounded history, one turn at a time per session. A failed LLM call leaves no half-finished tool exchange. Internal errors are logged and never shown to the guest. |

## Quick start

1. Start **LM Studio's local server on port 1235** with `gemma-4-26b-a4b-it` loaded.
2. Run:
   ```bash
   pip install -r requirements.txt
   uvicorn server:app --host 127.0.0.1 --port 8001
   ```
3. Open `http://localhost:8001`. The demo database is created on first start; `python setup_db.py` rebuilds it (and deletes all reservations).

Environment variables: `LLM_BASE_URL`, `LLM_MODEL`, `WHISPER_MODEL`, `RESTAURANT_DB_PATH`.

## Tests

21 deterministic tests run in a few seconds without a GPU: a scripted fake of LM Studio (`tests/fake_lmstudio.py`) replaces the model and the clock is frozen.

```bash
pip install -r requirements-dev.txt
python -m pytest
```

They cover the booking rules (hours, past dates, capacity overlaps, alternatives, cancellation), search and allergen filtering, deterministic demo data, session isolation, recovery from an LLM outage, hidden internal errors and the tool-loop limit.

## Project layout

| Path | Purpose |
|---|---|
| `server.py` | FastAPI endpoints: chat (text or audio), TTS, staff view, health |
| `restaurant/tools.py` | Search, menu, availability, booking and cancellation rules |
| `restaurant/registry.py` | Tool schemas for the model and a dispatcher that never raises |
| `restaurant/agent.py` | System prompt and the tool-calling loop |
| `restaurant/voice.py` | Speech-to-text and text-to-speech |
| `restaurant/data.py` | Schema and deterministic demo data (also exported to `restaurants_1000.csv`) |
| `index.html` | Single-page UI: conversation and staff view |

## Limitations

A local portfolio demo: no authentication on the staff view, sessions in memory, and text-to-speech sends reply text to an online service. A public deployment would need HTTPS, authentication, rate limits and persistent storage.
