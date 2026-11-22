"""API tests: the real FastAPI app and agent loop over HTTP, with a scripted fake LLM."""
import json

import httpx
import pytest

import server
from restaurant import agent, config, lmstudio, registry, tools


@pytest.fixture()
def client(app_url):
    with httpx.Client(base_url=app_url, timeout=30) as c:
        yield c


def chat(client, message):
    return client.post("/api/chat", data={"message": message}).json()


def first_restaurant():
    with tools.db() as conn:
        row = conn.execute("SELECT restaurant_id, opening_hours FROM restaurants WHERE opening_hours = '10:00-22:00' "
                           "ORDER BY restaurant_id LIMIT 1").fetchone()
    return row["restaurant_id"]


def test_booking_turn_runs_tools_and_shows_up_in_the_staff_view(client, fake_lm):
    rid = first_restaurant()
    fake_lm.script(("check_availability", {"restaurant_id": rid, "date": "2026-11-21", "time": "19:30", "guests": 2}),
                   ("book_table", {"restaurant_id": rid, "customer_name": "Alex", "date": "2026-11-21",
                                   "time": "19:30", "guests": 2}),
                   reply="Booked.")
    data = chat(client, "Yes, please book it")
    assert data["reply"] == "Booked."
    assert [(c["name"], c["status"]) for c in data["tool_calls"]] == [("check_availability", "success"),
                                                                      ("book_table", "success")]
    reservations = client.get("/api/reservations").json()["reservations"]
    assert reservations[0]["customer_name"] == "Alex" and reservations[0]["status"] == "confirmed"


def test_system_prompt_contains_today_and_all_tools(client, fake_lm):
    fake_lm.script(reply="Hi")
    chat(client, "hello")
    request = fake_lm.requests[-1]
    assert "Friday, 2026-11-20" in request["messages"][0]["content"]
    assert {t["function"]["name"] for t in request["tools"]} == {
        "search_restaurants", "get_menu", "check_availability", "book_table", "find_reservation", "update_reservation",
        "cancel_reservation"}


def test_conversations_are_isolated_per_session(app_url, fake_lm):
    with httpx.Client(base_url=app_url) as alice, httpx.Client(base_url=app_url) as bob:
        fake_lm.script(reply="Hi Alice")
        chat(alice, "My name is Alice and my phone is 0888 123 456")
        fake_lm.script(reply="Hi Bob")
        chat(bob, "Hello")
    sent_to_model = json.dumps(fake_lm.requests[-1]["messages"])
    assert "Alice" not in sent_to_model  # Bob's turn never sees Alice's conversation


def test_llm_outage_returns_a_safe_message_and_keeps_history_clean(client, fake_lm):
    fake_lm.fail_count = agent.llm_client.max_retries + 1
    data = chat(client, "Book a table")
    assert data["system_error"] is True
    assert "colleague" in data["reply"]
    assert "500" not in data["reply"] and "outage" not in data["reply"]  # no internal details
    history = next(iter(server.sessions.values()))["history"]
    assert history == []  # the failed turn left nothing behind

    fake_lm.script(reply="Back online")
    assert chat(client, "Hello again")["reply"] == "Back online"


def test_tool_crash_is_hidden_from_the_guest(client, fake_lm, monkeypatch):
    def broken(**_):
        raise RuntimeError("database file is locked at C:\\secret\\path")
    monkeypatch.setitem(registry.AVAILABLE_FUNCTIONS, "search_restaurants", broken)
    fake_lm.script(("search_restaurants", {"city": "Sofia"}), reply="Sorry, let me transfer you.")
    data = chat(client, "Find a restaurant")
    assert data["tool_calls"][0]["status"] == "system_error"
    tool_message = fake_lm.requests[-1]["messages"][-1]
    assert tool_message["role"] == "tool" and "secret" not in tool_message["content"]


def test_unknown_tool_and_bad_arguments_go_back_to_the_model(client, fake_lm):
    fake_lm.script(("teleport", {}), ("book_table", {"restaurant_id": "REST-0001"}), reply="Let me ask again.")
    data = chat(client, "Book")
    assert [c["status"] for c in data["tool_calls"]] == ["error", "error"]
    assert data["reply"] == "Let me ask again."


def test_tool_loop_is_capped(client, fake_lm):
    fake_lm.script_tool_loop("search_restaurants", {"city": "Sofia"}, rounds=config.MAX_TOOL_ROUNDS + 1)
    data = chat(client, "Search forever")
    assert len(data["tool_calls"]) == config.MAX_TOOL_ROUNDS + 1
    assert "colleague" in data["reply"]


def test_history_is_trimmed_only_at_user_turns():
    history = []
    for i in range(20):
        history += [{"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}"}]},
                    {"role": "tool", "tool_call_id": f"c{i}", "content": "{}"},
                    {"role": "assistant", "content": f"a{i}"}]
    trimmed = agent.trim_history(history, max_turns=5)
    assert trimmed[0] == {"role": "user", "content": "q15"}
    assert len(trimmed) == 20


def test_reset_and_empty_message(client, fake_lm):
    fake_lm.script(reply="Hi")
    chat(client, "hello")
    assert client.post("/api/reset").json()["status"] == "ok"
    assert next(iter(server.sessions.values()))["history"] == []
    assert client.post("/api/chat", data={"message": "   "}).status_code == 400


def test_health_reports_the_model(client):
    data = client.get("/api/health").json()
    assert data["llm"] == {"reachable": True, "model_available": True}


def test_reservation_codes_are_spelled_for_the_voice_only():
    from restaurant.voice import clean_for_speech
    assert clean_for_speech("Your code is R34LST. See you!") == "Your code is R, 3, 4, L, S, T. See you!"
    assert clean_for_speech("RADIOS and ROASTS stay words.") == "RADIOS and ROASTS stay words."


# --- Model selection ---------------------------------------------------------------------

def test_model_switch_unloads_the_current_model_first(client, fake_lm):
    result = client.post("/api/model", data={"model": "qwen3.8-27b"}).json()
    assert result == {"status": "success", "active": "qwen3.8-27b"}
    assert fake_lm.loaded == {"qwen3.8-27b"}  # only one model in GPU memory
    models = client.get("/api/models").json()
    assert models["active"] == "qwen3.8-27b"
    assert [m["id"] for m in models["models"]] == ["gemma-4-26b-a4b-it", "qwen3.8-27b"]
    assert client.post("/api/model", data={"model": "gpt-9"}).status_code == 400


def test_chat_uses_the_active_model(client, fake_lm):
    client.post("/api/model", data={"model": "qwen3.8-27b"})
    fake_lm.script(reply="Hi from Qwen")
    data = chat(client, "hello")
    assert data["model"] == "qwen3.8-27b"
    assert fake_lm.requests[-1]["model"] == "qwen3.8-27b"


def test_chat_waits_while_the_model_is_switching(client, fake_lm):
    lmstudio.model_state.update(switching=True, target="qwen3.8-27b")
    data = chat(client, "Маса за двама утре")
    assert data["system_error"] is True and "AI модел" in data["reply"]
    assert fake_lm.requests == []  # the LLM was not called



def test_chosen_speech_language_is_passed_to_transcription(client, fake_lm, monkeypatch):
    seen = {}

    def fake_transcribe(path, language=None):
        seen["language"] = language
        return "Маса за двама", "bg"

    monkeypatch.setattr(server, "transcribe_audio", fake_transcribe)
    fake_lm.script(reply="Добре.")
    data = client.post("/api/chat", data={"language": "bg"},
                       files={"audio": ("speech.webm", b"x" * 2000, "audio/webm")}).json()
    assert seen["language"] == "bg"
    assert data["user_text"] == "Маса за двама" and data["spoken_language"] == "bg"


def test_fixed_language_skips_whisper_language_detection(monkeypatch):
    import faster_whisper
    from restaurant import voice

    class FakeModel:
        def detect_language(self, *a, **k):
            raise AssertionError("detection must be skipped when the guest chose a language")

        def transcribe(self, audio, language, **kwargs):
            self.kwargs = {"language": language, **kwargs}
            return [type("Segment", (), {"text": " Здравейте "})()], None

    model = FakeModel()
    monkeypatch.setattr(voice, "get_whisper_model", lambda: model)
    monkeypatch.setattr(faster_whisper, "decode_audio", lambda path: "AUDIO")
    assert voice.transcribe_audio("x.webm", "bg") == ("Здравейте", "bg")
    assert model.kwargs["language"] == "bg" and "Ястия" in model.kwargs["initial_prompt"]
