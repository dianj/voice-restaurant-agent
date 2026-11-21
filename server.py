"""FastAPI app: chat (text or voice), text-to-speech, staff view of reservations and health check.

The logic lives in the `restaurant` package; this module wires it to HTTP. Run with: uvicorn server:app
"""
import asyncio
import json
import logging
import os
import tempfile
import uuid
from collections import OrderedDict

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, Response
from openai import APIConnectionError, APITimeoutError

from restaurant import config
from restaurant.agent import llm_client, run_llm_turn, trim_history
from restaurant.data import ensure_database
from restaurant.tools import list_reservations
from restaurant.voice import clean_for_speech, detect_text_language, synthesize_speech, transcribe_audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("restaurant")

app = FastAPI(title="Multilingual Voice Restaurant Agent")
ensure_database()

MAX_SESSIONS = 200
MAX_MESSAGE_CHARS = 1000
SESSION_COOKIE = "restaurant_session"
# session id -> {"history": [...], "lock": asyncio.Lock()}; the oldest sessions are dropped first.
sessions: "OrderedDict[str, dict]" = OrderedDict()

NOT_UNDERSTOOD = {
    "en": "Sorry, I didn't catch that. Could you say it again?",
    "bg": "Извинете, не ви чух добре. Бихте ли повторили?",
    "de": "Entschuldigung, das habe ich nicht verstanden. Könnten Sie es wiederholen?",
    "es": "Perdone, no le he entendido. ¿Podría repetirlo?",
    "fr": "Pardon, je n'ai pas bien compris. Pourriez-vous répéter ?",
    "it": "Mi scusi, non ho capito. Può ripetere?",
}
TECHNICAL_PROBLEM = {
    "en": "Sorry, I have a technical problem right now. Let me transfer you to a colleague.",
    "bg": "Извинете, в момента имам технически проблем. Ще ви свържа с колега.",
    "de": "Entschuldigung, ich habe gerade ein technisches Problem. Ich verbinde Sie mit einem Kollegen.",
    "es": "Lo siento, tengo un problema técnico. Le paso con un compañero.",
    "fr": "Désolé, j'ai un problème technique. Je vous passe un collègue.",
    "it": "Mi scusi, ho un problema tecnico. La passo a un collega.",
}


def get_session(request: Request):
    session_id = request.cookies.get(SESSION_COOKIE)
    if session_id in sessions:
        sessions.move_to_end(session_id)
    else:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {"history": [], "lock": asyncio.Lock()}
        while len(sessions) > MAX_SESSIONS:
            sessions.popitem(last=False)
    return session_id, sessions[session_id]


def with_cookie(payload: dict, session_id: str, status_code: int = 200) -> JSONResponse:
    response = JSONResponse(payload, status_code=status_code)
    response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="lax")
    return response


def tool_summary(call: dict) -> dict:
    try:
        status = json.loads(call["result"]).get("status", "unknown")
    except (json.JSONDecodeError, AttributeError):
        status = "unknown"
    return {"name": call["name"], "arguments": call["arguments"], "status": status}


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open(config.INDEX_PATH, encoding="utf-8") as f:
        return f.read()


@app.post("/api/chat")
async def api_chat(request: Request, message: str = Form(""), audio: UploadFile = File(None)):
    session_id, session = get_session(request)
    spoken_language = None
    text = message.strip()

    if audio is not None and audio.filename:
        raw = await audio.read()
        if len(raw) > config.MAX_AUDIO_BYTES:
            return with_cookie({"error": "The recording is too long."}, session_id, 413)
        suffix = os.path.splitext(audio.filename)[1] or ".webm"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            tmp.write(raw)
            tmp.close()
            text, spoken_language = await run_in_threadpool(transcribe_audio, tmp.name)
        except Exception:
            log.exception("Speech recognition failed")
            return with_cookie({"error": "Speech recognition failed. Please try again or type your message."},
                               session_id, 500)
        finally:
            os.remove(tmp.name)
        if len(text) < 2:
            reply = NOT_UNDERSTOOD.get(spoken_language, NOT_UNDERSTOOD["en"])
            return with_cookie({"user_text": "", "spoken_language": spoken_language, "reply": reply,
                                "reply_language": spoken_language, "tool_calls": []}, session_id)

    if not text:
        return with_cookie({"error": "Type a message or record one."}, session_id, 400)
    text = text[:MAX_MESSAGE_CHARS]

    async with session["lock"]:  # one turn at a time per conversation
        # Work on a copy, so a failed turn never leaves a half-finished tool exchange in the session.
        turn_history = session["history"] + [{"role": "user", "content": text}]
        tool_calls = []
        try:
            msg = await run_in_threadpool(run_llm_turn, turn_history, tool_calls, spoken_language)
            if msg.tool_calls:
                reply = TECHNICAL_PROBLEM.get(spoken_language or detect_text_language(text), TECHNICAL_PROBLEM["en"])
                log.warning("Tool loop limit reached")
            else:
                reply = (msg.content or "").strip() or NOT_UNDERSTOOD["en"]
            turn_history.append({"role": "assistant", "content": reply})
            session["history"] = trim_history(turn_history)
            failed = False
        except (APIConnectionError, APITimeoutError):
            log.error("LM Studio is not reachable at %s", config.LLM_BASE_URL)
            reply, failed = None, True
        except Exception:
            log.exception("Chat turn failed")
            reply, failed = None, True

    reply_language = detect_text_language(reply) if reply else (spoken_language or detect_text_language(text))
    payload = {"user_text": text, "spoken_language": spoken_language, "reply_language": reply_language,
               "tool_calls": [tool_summary(c) for c in tool_calls]}
    if failed:
        payload.update(reply=TECHNICAL_PROBLEM.get(reply_language, TECHNICAL_PROBLEM["en"]), system_error=True)
    else:
        payload["reply"] = reply
    return with_cookie(payload, session_id)


@app.post("/api/tts")
async def api_tts(text: str = Form(...), language: str = Form(None), gender: str = Form(config.DEFAULT_VOICE_GENDER)):
    spoken = clean_for_speech(text)
    if not spoken:
        return JSONResponse({"error": "Nothing to speak."}, status_code=400)
    if language not in config.SUPPORTED_LANGUAGES:
        language = detect_text_language(spoken)
    audio, error = await synthesize_speech(spoken, language, gender)
    if audio:
        return Response(content=audio, media_type="audio/mpeg")
    log.warning(error)
    return JSONResponse({"error": "Voice reply is not available right now."}, status_code=502)


@app.post("/api/reset")
async def api_reset(request: Request):
    session_id, session = get_session(request)
    session["history"] = []
    return with_cookie({"status": "ok"}, session_id)


@app.get("/api/reservations")
def api_reservations():
    return {"reservations": list_reservations()}


@app.get("/api/health")
def api_health():
    try:
        models = [m.id for m in llm_client.with_options(timeout=3, max_retries=0).models.list().data]
        llm = {"reachable": True, "model_available": config.MODEL_NAME in models}
    except Exception:
        llm = {"reachable": False, "model_available": False}
    return {"llm": llm, "model": config.MODEL_NAME, "llm_base_url": config.LLM_BASE_URL}
