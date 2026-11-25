"""FastAPI app: chat (text or voice), text-to-speech, staff view of reservations, model switching and health.

The logic lives in the `restaurant` package; this module wires it to HTTP. Run with: uvicorn server:app
"""
import asyncio
import json
import logging
import os
import secrets
import tempfile
import threading
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Header, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, Response
from openai import APIConnectionError, APITimeoutError

from restaurant import config, ratelimit
from restaurant.agent import llm_client, run_llm_turn, trim_history
from restaurant.data import ensure_database
from restaurant.lmstudio import ensure_default_model_on_startup, list_available_models, model_state, switch_model
from restaurant.tools import list_requests, list_reservations
from restaurant.voice import AudioTooLong, clean_for_speech, detect_text_language, synthesize_speech, transcribe_audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("restaurant")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the default model in the background so the page is available while LM Studio loads it.
    threading.Thread(target=ensure_default_model_on_startup, daemon=True).start()
    yield


app = FastAPI(title="Multilingual Voice Restaurant Agent", lifespan=lifespan)
ensure_database()

MAX_SESSIONS = 200
MAX_MESSAGE_CHARS = 1000
SESSION_COOKIE = "restaurant_session"
# session id -> {"history": [...], "lock": asyncio.Lock()}; the oldest sessions are dropped first.
sessions: "OrderedDict[str, dict]" = OrderedDict()
# Turns waiting for the LLM right now; a model switch would unload the model they are using.
active_turns = 0

NOT_UNDERSTOOD = {
    "en": "Sorry, I didn't catch that. Could you say it again?",
    "bg": "Извинете, не ви чух добре. Бихте ли повторили?",
    "de": "Entschuldigung, das habe ich nicht verstanden. Könnten Sie es wiederholen?",
    "es": "Perdone, no le he entendido. ¿Podría repetirlo?",
    "fr": "Pardon, je n'ai pas bien compris. Pourriez-vous répéter ?",
    "it": "Mi scusi, non ho capito. Può ripetere?",
}
MODEL_SWITCHING = {
    "en": "One moment please, the assistant is switching to another AI model. Try again in a minute.",
    "bg": "Момент, моля: асистентът превключва към друг AI модел. Опитайте отново след минута.",
}
TECHNICAL_PROBLEM = {
    # Nothing is promised here: when the model is down, no one can take a callback request either.
    "en": "Sorry, I have a technical problem right now. Please try again in a few minutes.",
    "bg": "Извинете, в момента имам технически проблем. Моля, опитайте отново след няколко минути.",
    "de": "Entschuldigung, ich habe gerade ein technisches Problem. Bitte versuchen Sie es in ein paar Minuten erneut.",
    "es": "Lo siento, tengo un problema técnico. Por favor, inténtelo de nuevo en unos minutos.",
    "fr": "Désolé, j'ai un problème technique. Veuillez réessayer dans quelques minutes.",
    "it": "Mi scusi, ho un problema tecnico. Riprovi tra qualche minuto.",
}


limiters = {"chat": ratelimit.RateLimiter(config.CHAT_PER_MINUTE),
            "staff": ratelimit.RateLimiter(config.STAFF_PER_MINUTE)}


def client_address(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def too_many(limiter: ratelimit.RateLimiter) -> JSONResponse:
    return JSONResponse({"error": "Too many requests. Please wait a moment."}, status_code=429,
                        headers={"Retry-After": str(limiter.retry_after())})


def rate_limited(request: Request) -> JSONResponse | None:
    """429 when this address has used up its chat and speech requests for now; None otherwise."""
    limiter = limiters["chat"]
    return None if limiter.allow(client_address(request)) else too_many(limiter)


def get_session(request: Request):
    """This browser's session; a new one only for a real turn (a chat message), which is rate limited."""
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
async def api_chat(request: Request, message: str = Form(""), audio: UploadFile = File(None),
                   language: str = Form("auto")):
    if refused := rate_limited(request):
        return refused
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
            # The guest can fix the speech language in the UI; "auto" lets Whisper detect it.
            text, spoken_language = await run_in_threadpool(transcribe_audio, tmp.name, language)
        except AudioTooLong as exc:
            log.info("Voice message refused: %s", exc)
            return with_cookie({"error": f"The recording is too long. Please keep it under {config.MAX_AUDIO_SECONDS} "
                                         "seconds, or type your message."}, session_id, 413)
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
    if model_state["switching"]:
        language = spoken_language or detect_text_language(text)
        return with_cookie({"user_text": text, "spoken_language": spoken_language, "reply_language": language,
                            "reply": MODEL_SWITCHING.get(language, MODEL_SWITCHING["en"]), "tool_calls": [],
                            "system_error": True}, session_id)
    active_model = model_state["active"]

    global active_turns
    async with session["lock"]:  # one turn at a time per conversation
        active_turns += 1
        # Work on a copy, so a failed turn never leaves a half-finished tool exchange in the session.
        turn_history = session["history"] + [{"role": "user", "content": text}]
        tool_calls = []
        try:
            msg = await run_in_threadpool(run_llm_turn, turn_history, tool_calls, spoken_language, active_model)
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
        finally:
            active_turns -= 1

    reply_language = detect_text_language(reply) if reply else (spoken_language or detect_text_language(text))
    payload = {"user_text": text, "spoken_language": spoken_language, "reply_language": reply_language,
               "tool_calls": [tool_summary(c) for c in tool_calls], "model": active_model}
    if failed:
        payload.update(reply=TECHNICAL_PROBLEM.get(reply_language, TECHNICAL_PROBLEM["en"]), system_error=True)
    else:
        payload["reply"] = reply
    return with_cookie(payload, session_id)


@app.post("/api/tts")
async def api_tts(request: Request, text: str = Form(...), language: str = Form(None),
                  gender: str = Form(config.DEFAULT_VOICE_GENDER)):
    # edge-tts is free, so the text stays open; the limit keeps one address from flooding it.
    if refused := rate_limited(request):
        return refused
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
    session_id = request.cookies.get(SESSION_COOKIE)
    if session_id not in sessions:          # nothing to reset; the first message makes the session
        return Response(status_code=204)
    sessions[session_id]["history"] = []
    return with_cookie({"status": "ok"}, session_id)


def staff_refusal(request: Request, token: str | None) -> JSONResponse | None:
    """None for the right staff token. Wrong tokens are counted per address, so it cannot be guessed quickly."""
    limiter, address = limiters["staff"], client_address(request)
    if limiter.blocked(address):
        return too_many(limiter)
    if token and secrets.compare_digest(token.encode(), config.STAFF_TOKEN.encode()):
        return None
    limiter.allow(address)
    return JSONResponse({"error": "Staff token required."}, status_code=401)


@app.get("/api/reservations")
def api_reservations(request: Request, x_staff_token: str | None = Header(None)):
    """The staff view: names, phone numbers, allergy notes and callback contacts, so only with the staff token."""
    if refused := staff_refusal(request, x_staff_token):
        return refused
    return {"reservations": list_reservations(), "requests": list_requests()}


@app.get("/api/health")
def api_health():
    active = model_state["active"]
    try:
        models = [m.id for m in llm_client.with_options(timeout=3, max_retries=0).models.list().data]
        llm = {"reachable": True, "model_available": active in models}
    except Exception:
        llm = {"reachable": False, "model_available": False}
    return {"llm": llm, "model": active, "switching": model_state["switching"]}


@app.get("/api/models")
def api_models():
    return {"models": list_available_models(), "active": model_state["active"],
            "switching": model_state["switching"], "target": model_state["target"]}


@app.post("/api/model")
def api_select_model(model: str = Form(...)):
    # Sync endpoint: FastAPI runs it in a worker thread, so the long LM Studio load does not block the server.
    if model not in {model_id for model_id, _ in config.ALLOWED_MODELS}:
        return JSONResponse({"status": "error", "message": f"Unknown model '{model}'."}, status_code=400)
    if model_state["switching"]:
        return JSONResponse({"status": "error", "message": "Another model switch is already in progress."},
                            status_code=409)
    if active_turns:
        return JSONResponse({"status": "error", "message": "A guest is waiting for a reply; try again in a moment."},
                            status_code=409)
    result = switch_model(model)
    return JSONResponse(result, status_code=200 if result["status"] == "success" else 500)
