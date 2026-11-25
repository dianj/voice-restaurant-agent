"""Who can reach what (review T22.A5–A7, T25.S5–S6): staff data behind a token, sessions that strangers cannot push
out, request limits per address, no allergy notes for anyone with a code, and a cap on the length of a recording."""
import json

import pytest
from fastapi.testclient import TestClient

import server
from restaurant import config, ratelimit, tools, voice

STAFF = {"X-Staff-Token": config.STAFF_TOKEN}


def test_staff_data_needs_the_staff_token():
    """Names, phones, allergy notes and callback contacts were open to anyone who opened /api/reservations."""
    tools.request_callback("event", "Ivan Petrov", "0888 123 456", "Children's party for 25", guests=25)
    guest = TestClient(server.app)
    assert guest.get("/api/reservations").status_code == 401
    assert guest.get("/api/reservations", headers={"X-Staff-Token": "guess"}).status_code == 401
    staff = guest.get("/api/reservations", headers=STAFF)
    assert staff.status_code == 200 and staff.json()["requests"][0]["contact"] == "0888 123 456"


def test_a_token_is_made_when_none_is_set(monkeypatch, capsys):
    monkeypatch.delenv("STAFF_TOKEN", raising=False)
    first, second = config.load_staff_token(), config.load_staff_token()
    assert first != second and len(first) >= 16
    assert first in capsys.readouterr().out                     # printed once, for the person who started it
    monkeypatch.setenv("STAFF_TOKEN", "my-own-token")
    assert config.load_staff_token() == "my-own-token"


def test_find_reservation_does_not_show_the_notes():
    """Allergies are health data: a code alone must not reveal them."""
    with tools.db() as conn:
        rid = conn.execute("SELECT restaurant_id FROM restaurants WHERE opening_hours = '10:00-22:00' "
                           "ORDER BY restaurant_id LIMIT 1").fetchone()[0]
    code = json.loads(tools.book_table(rid, "Maria", "2026-11-21", "19:00", 2, notes="nut allergy"))["reservation_code"]
    found = json.loads(tools.find_reservation(code))
    assert "notes" not in found["reservation"] and "nut allergy" not in json.dumps(found)
    assert found["reservation"]["has_notes"] is True


def test_cookieless_resets_do_not_push_a_guest_out(fake_lm):
    guest = TestClient(server.app)
    fake_lm.script(reply="Hello!")
    guest.post("/api/chat", data={"message": "Hi"})
    cookie = guest.cookies.get(server.SESSION_COOKIE)
    stranger = TestClient(server.app)
    for _ in range(server.MAX_SESSIONS + 50):
        stranger.cookies.clear()
        assert stranger.post("/api/reset").status_code == 204
    assert cookie in server.sessions and len(server.sessions) == 1


def test_the_61st_chat_in_a_minute_from_one_address_gets_429(fake_lm, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(ratelimit, "monotonic", lambda: clock[0])
    client = TestClient(server.app)
    for _ in range(config.CHAT_PER_MINUTE):
        fake_lm.script(reply="ok")
        assert client.post("/api/chat", data={"message": "Hi"}).status_code == 200
    refused = client.post("/api/chat", data={"message": "Hi"})
    assert refused.status_code == 429 and "retry-after" in refused.headers
    assert client.post("/api/tts", data={"text": "Hello"}).status_code == 429     # speech shares the chat bucket
    clock[0] += 2
    fake_lm.script(reply="ok")
    assert client.post("/api/chat", data={"message": "Hi"}).status_code == 200


def test_guessing_the_staff_token_is_limited(monkeypatch):
    monkeypatch.setattr(ratelimit, "monotonic", lambda: 5000.0)
    client = TestClient(server.app)
    codes = [client.get("/api/reservations", headers={"X-Staff-Token": "guess"}).status_code
             for _ in range(config.STAFF_PER_MINUTE + 1)]
    assert codes[:-1] == [401] * config.STAFF_PER_MINUTE and codes[-1] == 429


def test_the_bucket_is_per_address():
    bucket = ratelimit.RateLimiter(per_minute=2)
    assert bucket.allow("a", now=0) and bucket.allow("a", now=0) and not bucket.allow("a", now=0)
    assert bucket.allow("b", now=0)
    assert bucket.allow("a", now=30)


def test_health_does_not_show_the_llm_address():
    health = TestClient(server.app).get("/api/health").json()
    assert "llm_base_url" not in health and config.LLM_BASE_URL not in json.dumps(health)


def silent_wav(path, seconds: float):
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(bytes(2) * int(16000 * seconds))
    return str(path)


def test_a_recording_over_a_minute_is_refused_after_decoding(tmp_path, monkeypatch):
    """T25.S6: 15 MB of Opus is about an hour of speech for Whisper on the CPU. Measured on the decoded audio (a real
    90 s file here); the model is neither loaded nor asked."""
    monkeypatch.setattr(voice, "get_whisper_model", lambda: pytest.fail("Whisper must not run"))
    with pytest.raises(voice.AudioTooLong):
        voice.transcribe_audio(silent_wav(tmp_path / "long.wav", 90), "en")


def test_a_too_long_recording_gets_a_friendly_413(monkeypatch):
    def too_long(path, language):
        raise voice.AudioTooLong(95.0)

    monkeypatch.setattr(server, "transcribe_audio", too_long)
    response = TestClient(server.app).post("/api/chat", files={"audio": ("clip.webm", b"x" * 100, "audio/webm")})
    assert response.status_code == 413 and "60 seconds" in response.json()["error"]
