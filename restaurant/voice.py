"""Voice: local speech-to-text (faster-whisper) and text-to-speech (edge-tts, online)."""
import asyncio
import re
import threading

from . import config

_whisper_model = None
_whisper_lock = threading.Lock()


def get_whisper_model():
    """Load Whisper lazily on the first voice request so the server starts fast."""
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            from faster_whisper import WhisperModel
            _whisper_model = WhisperModel(config.WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        return _whisper_model


def pick_supported_language(model, audio) -> str:
    """Detect the spoken language, restricted to the supported ones.

    Whisper often labels short Bulgarian clips as Russian or Macedonian, so instead of trusting the
    top label we compare only the probabilities of the languages the agent supports.
    """
    try:
        _, _, all_probs = model.detect_language(audio, vad_filter=True)
        probs = dict(all_probs)
        return max(config.SUPPORTED_LANGUAGES, key=lambda lang: probs.get(lang, 0.0))
    except Exception:
        return config.PRIMARY_LANGUAGE


def transcribe_audio(path: str):
    from faster_whisper import decode_audio
    model = get_whisper_model()
    audio = decode_audio(path)
    language = pick_supported_language(model, audio)
    segments, _ = model.transcribe(audio, language=language, vad_filter=True, beam_size=5,
                                   initial_prompt=config.WHISPER_PROMPTS[language])
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return text, language


# Short, frequent words per language, used to pick the TTS voice for a typed (not spoken) conversation.
STOPWORDS = {
    "en": {"the", "and", "you", "for", "your", "table", "is", "at", "have", "would", "a", "to"},
    "de": {"der", "die", "das", "und", "ist", "ich", "sie", "ein", "eine", "für", "tisch", "uhr", "nicht", "mit"},
    "es": {"el", "la", "los", "las", "que", "para", "una", "mesa", "con", "por", "es", "su", "y"},
    "fr": {"le", "la", "les", "et", "est", "une", "pour", "vous", "table", "avec", "des", "pas", "votre"},
    "it": {"il", "la", "che", "per", "una", "tavolo", "con", "è", "di", "sono", "alle", "gli", "non"},
}


def detect_text_language(text: str) -> str:
    cyrillic = len(re.findall(r"[Ѐ-ӿ]", text))
    latin = len(re.findall(r"[A-Za-zÀ-ÿ]", text))
    if cyrillic and cyrillic >= latin * 0.3:
        return "bg"
    words = re.findall(r"[a-zà-ÿ]+", text.lower())
    scores = {lang: sum(w in stop for w in words) for lang, stop in STOPWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else config.PRIMARY_LANGUAGE


RESERVATION_CODE = re.compile(r"\bR-([A-Z0-9]{5})\b")


def spell_codes(text: str) -> str:
    """'R-RM952' -> 'R, R, M, 9, 5, 2', so the voice reads a reservation code character by character."""
    return RESERVATION_CODE.sub(lambda m: ", ".join("R" + m.group(1)), text)


def clean_for_speech(text: str) -> str:
    text = spell_codes(text)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s{0,3}(#{1,6}|>|[-*•])\s+", "", text, flags=re.M)
    text = re.sub(r"[*`_#]", "", text)
    text = re.sub(r"https?://\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()[:config.MAX_TTS_CHARS]


async def synthesize_speech(text: str, language: str, gender: str):
    """Text -> MP3 bytes. Returns (audio, None) or (None, error description)."""
    import edge_tts
    voices = config.TTS_VOICES.get(gender, config.TTS_VOICES[config.DEFAULT_VOICE_GENDER])
    voice = voices.get(language, voices[config.PRIMARY_LANGUAGE])

    async def synthesize() -> bytes:
        audio = bytearray()
        async for chunk in edge_tts.Communicate(text, voice).stream():
            if chunk["type"] == "audio":
                audio.extend(chunk["data"])
        return bytes(audio)

    last_error = None
    # The online service sometimes answers "No audio was received", so retry with a short backoff.
    for attempt in range(config.TTS_ATTEMPTS):
        try:
            audio = await asyncio.wait_for(synthesize(), timeout=config.TTS_ATTEMPT_TIMEOUT)
            if audio:
                return audio, None
            last_error = "empty audio"
        except asyncio.TimeoutError:
            last_error = f"timeout after {config.TTS_ATTEMPT_TIMEOUT}s"
        except Exception as e:
            last_error = str(e)
        await asyncio.sleep(0.6 * (attempt + 1))
    return None, f"Text-to-speech failed after {config.TTS_ATTEMPTS} attempts: {last_error}"
