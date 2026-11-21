import os
import json
import sqlite3
import tempfile
import edge_tts
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from faster_whisper import WhisperModel

app = FastAPI(title="Multilingual Voice Restaurant AI Agent")
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

llm_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
MODEL_NAME = "gemma-4-26b-a4b-it"
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

TTS_VOICES = {
    "bg": "bg-BG-BorislavNeural",
    "en": "en-US-AndrewNeural",
    "de": "de-DE-ConradNeural",
    "es": "es-ES-AlvaroNeural",
    "fr": "fr-FR-HenriNeural",
    "it": "it-IT-DiegoNeural"
}

def search_restaurants(cuisine: str = None, location: str = None):
    try:
        conn = sqlite3.connect("restaurant_agent.db")
        conn.row_factory = sqlite3.Row
        query = "SELECT restaurant_id, name, location, cuisine, price_range, rating, opening_hours, seating_capacity FROM restaurants WHERE 1=1"
        params = []
        if cuisine:
            query += " AND cuisine LIKE ?"
            params.append(f"%{cuisine}%")
        if location:
            query += " AND location LIKE ?"
            params.append(f"%{location}%")
        query += " LIMIT 5"
        rows = [dict(r) for r in conn.execute(query, params).fetchall()]
        conn.close()
        return json.dumps(rows if rows else {"error": "No matching restaurants found."})
    except Exception as e:
        return json.dumps({"system_error": f"Database unavailable: {str(e)}"})

def get_menu_and_allergens(restaurant_id: str):
    try:
        conn = sqlite3.connect("restaurant_agent.db")
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(
            "SELECT dish_name, ingredients, allergens, price FROM menu_items WHERE restaurant_id = ?",
            (restaurant_id.upper(),)
        ).fetchall()]
        conn.close()
        return json.dumps(rows if rows else {"info": "No specific menu items listed for this ID. Suggest REST-0001, REST-0002, or REST-0003 for full menu demo."})
    except Exception as e:
        return json.dumps({"system_error": str(e)})

def check_and_book_table(restaurant_id: str, customer_name: str, date: str, time: str, guests: int):
    try:
        conn = sqlite3.connect("restaurant_agent.db")
        cur = conn.cursor()
        rest = cur.execute("SELECT name, seating_capacity, opening_hours FROM restaurants WHERE restaurant_id = ?", (restaurant_id.upper(),)).fetchone()
        if not rest:
            conn.close()
            return json.dumps({"status": "failed", "reason": "Restaurant ID not found."})
        
        name, capacity, hours = rest
        if guests > capacity:
            conn.close()
            return json.dumps({"status": "failed", "reason": f"Requested {guests} guests exceeds max capacity of {capacity}."})

        cur.execute(
            "INSERT INTO reservations (restaurant_id, customer_name, date, time, guests, status) VALUES (?, ?, ?, ?, ?, 'CONFIRMED')",
            (restaurant_id.upper(), customer_name, date, time, guests)
        )
        res_id = cur.lastrowid
        conn.commit()
        conn.close()
        return json.dumps({"status": "confirmed", "reservation_id": res_id, "restaurant": name, "date": date, "time": time, "guests": guests})
    except Exception as e:
        return json.dumps({"system_error": str(e)})

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_restaurants",
            "description": "Search restaurants by cuisine type, location, or get opening hours and ratings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cuisine": {"type": "string", "description": "e.g. Italian, Bulgarian, Japanese"},
                    "location": {"type": "string", "description": "e.g. Sofia Center, Lozenets, London Soho"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_menu_and_allergens",
            "description": "Get menu dishes, ingredients, and allergy information for a restaurant ID (e.g. REST-0001).",
            "parameters": {
                "type": "object",
                "properties": {
                    "restaurant_id": {"type": "string", "description": "Unique restaurant ID like REST-0001"}
                },
                "required": ["restaurant_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_and_book_table",
            "description": "Check table availability and book a restaurant reservation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "restaurant_id": {"type": "string"},
                    "customer_name": {"type": "string"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "time": {"type": "string", "description": "HH:MM"},
                    "guests": {"type": "integer"}
                },
                "required": ["restaurant_id", "customer_name", "date", "time", "guests"]
            }
        }
    }
]

AVAILABLE_FUNCTIONS = {
    "search_restaurants": search_restaurants,
    "get_menu_and_allergens": get_menu_and_allergens,
    "check_and_book_table": check_and_book_table
}

conversation_history = [
    {
        "role": "system",
        "content": (
            "You are a polite, helpful phone AI concierge for a restaurant network. "
            "CRITICAL: Always reply in the EXACT SAME LANGUAGE the user spoke in (e.g., Bulgarian, English, German). "
            "Keep answers concise, natural for voice conversation, and free of markdown symbols like asterisks or bullet points. "
            "If a tool returns an error or no availability, apologize naturally and suggest an alternative. "
            "If system_error occurs, politely offer to transfer the call to a human operator."
        )
    }
]

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/voice-chat")
async def voice_chat(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    segments, info = whisper_model.transcribe(tmp_path)
    os.remove(tmp_path)
    
    user_text = " ".join([s.text for s in segments]).strip()
    detected_lang = info.language if info.language in TTS_VOICES else "en"

    if not user_text or len(user_text) < 2:
        fallback_msg = "Извинете, не ви чух добре. Бихте ли повторили?" if detected_lang == "bg" else "Sorry, I didn't catch that. Could you please repeat?"
        audio_url = await synthesize_speech(fallback_msg, detected_lang)
        return JSONResponse({"user_text": "[Unintelligible audio]", "bot_text": fallback_msg, "tool_used": "None (STT Fallback)", "audio_url": audio_url})

    conversation_history.append({"role": "user", "content": user_text})
    tool_used = "None"

    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=conversation_history,
            tools=TOOLS_SCHEMA
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            conversation_history.append(msg)
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                tool_used = f"{fn_name}({fn_args})"
                
                result = AVAILABLE_FUNCTIONS[fn_name](**fn_args)
                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

            final_res = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=conversation_history
            )
            bot_text = final_res.choices[0].message.content
        else:
            bot_text = msg.content

    except Exception as e:
        bot_text = "В момента имам технически проблем. Свързвам ви с оператор." if detected_lang == "bg" else "I'm experiencing a technical issue. Transferring you to a human agent."
        tool_used = f"System Error: {str(e)}"

    conversation_history.append({"role": "assistant", "content": bot_text})
    audio_url = await synthesize_speech(bot_text, detected_lang)

    return JSONResponse({
        "user_text": user_text,
        "detected_lang": detected_lang,
        "tool_used": tool_used,
        "bot_text": bot_text,
        "audio_url": audio_url
    })

async def synthesize_speech(text: str, lang: str) -> str:
    voice = TTS_VOICES.get(lang, "en-US-AndrewNeural")
    output_file = "static/response.mp3"
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)
    return f"/{output_file}?t={os.urandom(4).hex()}"
