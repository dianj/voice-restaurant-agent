"""Voice helpers with fake Whisper and edge-tts objects: no model download, no network."""
import asyncio
from asyncio import sleep as real_sleep  # kept before the backoff is patched out
from concurrent.futures import ThreadPoolExecutor

import edge_tts
import faster_whisper
import pytest

from restaurant import config, voice


class FakeWhisper:
    def __init__(self, probs=None, fail_detection=False, text=" A table for two "):
        self.probs, self.fail_detection, self.text = probs or {}, fail_detection, text
        self.transcribe_kwargs = None

    def detect_language(self, audio, **kwargs):
        if self.fail_detection:
            raise RuntimeError("no speech")
        return "xx", 1.0, list(self.probs.items())

    def transcribe(self, audio, language, **kwargs):
        self.transcribe_kwargs = {"language": language, **kwargs}
        return [type("Segment", (), {"text": part})() for part in self.text.split("|")], None


def test_language_detection_only_compares_supported_languages():
    # Short Bulgarian clips often come back as Russian or Macedonian; only supported languages are compared.
    assert voice.pick_supported_language(FakeWhisper({"ru": 0.5, "mk": 0.2, "bg": 0.2, "en": 0.1}), "A") == "bg"
    assert voice.pick_supported_language(FakeWhisper({"pt": 0.6, "es": 0.3, "it": 0.1}), "A") == "es"


def test_language_detection_failure_falls_back_to_the_primary_language():
    assert voice.pick_supported_language(FakeWhisper(fail_detection=True), "A") == config.PRIMARY_LANGUAGE


def test_transcription_detects_the_language_when_the_guest_did_not_choose(monkeypatch):
    model = FakeWhisper({"de": 0.8, "en": 0.2}, text=" Einen Tisch | für zwei ")
    monkeypatch.setattr(voice, "get_whisper_model", lambda: model)
    monkeypatch.setattr(faster_whisper, "decode_audio", lambda path: "AUDIO")
    assert voice.transcribe_audio("clip.webm", "auto") == ("Einen Tisch für zwei", "de")
    assert model.transcribe_kwargs["initial_prompt"] == config.WHISPER_PROMPTS["de"]


def test_whisper_is_loaded_once_and_only_when_first_needed(monkeypatch):
    created = []
    monkeypatch.setattr(faster_whisper, "WhisperModel", lambda *args, **kwargs: created.append(kwargs) or object())
    monkeypatch.setattr(voice, "_whisper_model", None)
    first = voice.get_whisper_model()
    assert voice.get_whisper_model() is first
    assert created == [{"device": "cpu", "compute_type": "int8"}]  # CPU: the GPU is reserved for the LLM


@pytest.mark.parametrize("text, language", [
    ("Your table is booked for tomorrow at 19:30.", "en"),
    ("Масата е запазена за утре в 19:30.", "bg"),
    ("Ich habe einen Tisch für Sie und die Uhrzeit ist frei.", "de"),
    ("La mesa para dos está libre por la noche.", "es"),
    ("Votre table pour deux est réservée avec vue.", "fr"),
    ("Il tavolo per due è libero alle 20.", "it"),
    ("19:30", "en"),  # no words to go by: the primary language
])
def test_reply_language_picks_the_tts_voice_for_typed_conversations(text, language):
    assert voice.detect_text_language(text) == language


@pytest.mark.parametrize("text, language", [
    ("Lo siento, no tenemos restaurantes en París.", "es"),
    ("Merci, à demain !", "fr"),
    ("Grazie mille, a domani.", "it"),
    ("¿Algo más?", "es"),
])
def test_short_spanish_french_and_italian_replies_get_their_own_voice(text, language):
    """Review T22.A8: these short replies scored no stop words and were read with the English voice."""
    assert voice.detect_text_language(text) == language


def test_a_reply_with_too_few_words_keeps_the_guests_language():
    assert voice.detect_text_language("R34LST, 19:30.", fallback="it") == "it"
    assert voice.detect_text_language("OK!", fallback="fr") == "fr"
    assert voice.detect_text_language("Ich habe einen Tisch für Sie.", fallback="en") == "de"   # a clear reply wins


def test_the_reply_voice_follows_the_guest_when_the_reply_is_short(fake_lm):
    from fastapi.testclient import TestClient

    import server
    client = TestClient(server.app)

    def reply_language(message):
        return client.post("/api/chat", data={"message": message}).json()["reply_language"]

    fake_lm.script(reply="Vale.")
    assert reply_language("Hola, ¿tienen una mesa para dos esta noche?") == "es"
    fake_lm.script(reply="R34LST, 19:30.")
    assert reply_language("Sí, gracias.") == "es"                  # the session remembers the guest's language


def test_markdown_and_links_are_not_read_aloud_and_length_is_capped():
    text = "**Marble Hearth** has a table, see [the menu](https://x.io).\n- Code R34LST\nhttps://x.io"
    assert voice.clean_for_speech(text) == "Marble Hearth has a table, see the menu. Code R, 3, 4, L, S, T"
    assert len(voice.clean_for_speech("word " * 1000)) == config.MAX_TTS_CHARS


# --- text-to-speech ---------------------------------------------------------------------

def fake_communicate(plan, voices_used):
    """edge_tts.Communicate stand-in: each attempt takes the next behaviour from `plan`."""
    class Communicate:
        def __init__(self, text, voice):
            voices_used.append(voice)
            self.behaviour = plan.pop(0)

        async def stream(self):
            if self.behaviour == "error":
                raise ConnectionError("No audio was received")
            if self.behaviour == "hang":
                await real_sleep(5)
            if self.behaviour == "audio":
                yield {"type": "WordBoundary"}
                yield {"type": "audio", "data": b"MP3"}
    return Communicate


@pytest.fixture
def no_backoff(monkeypatch):
    monkeypatch.setattr(voice.asyncio, "sleep", lambda seconds: real_sleep(min(seconds, 0.01)))


def run_async(coroutine):
    """asyncio.run in its own thread: the Playwright tests keep an event loop running in the main thread."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()


def speak(monkeypatch, plan, language="en", gender="female"):
    voices_used = []
    monkeypatch.setattr(edge_tts, "Communicate", fake_communicate(list(plan), voices_used))
    return run_async(voice.synthesize_speech("Your table is booked.", language, gender)), voices_used


def test_tts_uses_the_voice_for_the_language_and_gender(monkeypatch, no_backoff):
    (audio, error), voices_used = speak(monkeypatch, ["audio"], "it", "male")
    assert (audio, error) == (b"MP3", None)
    assert voices_used == [config.TTS_VOICES["male"]["it"]]


def test_tts_falls_back_to_the_primary_language_and_default_gender(monkeypatch, no_backoff):
    (audio, _), voices_used = speak(monkeypatch, ["audio"], "pt", "robot")
    assert audio == b"MP3"
    assert voices_used == [config.TTS_VOICES[config.DEFAULT_VOICE_GENDER][config.PRIMARY_LANGUAGE]]


def test_tts_retries_after_an_error_and_an_empty_answer(monkeypatch, no_backoff):
    (audio, error), voices_used = speak(monkeypatch, ["error", "empty", "audio"])
    assert (audio, error) == (b"MP3", None)
    assert len(voices_used) == 3


def test_tts_gives_up_with_a_reason_after_all_attempts(monkeypatch, no_backoff):
    monkeypatch.setattr(config, "TTS_ATTEMPT_TIMEOUT", 0.05)
    (audio, error), _ = speak(monkeypatch, ["error", "empty", "hang"])
    assert audio is None
    assert error == f"Text-to-speech failed after {config.TTS_ATTEMPTS} attempts: timeout after 0.05s"
