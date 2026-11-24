"""The "wants a person" gate: on every turn, in parallel with the agent, a calibrated check whether the guest asks for
a human. Inverted (p(yes) >= 0.5 is enough). The code cannot create a callback by itself (it needs the guest's name
and contact), so when the agent did not call request_callback, it is called once more with an instruction to start
the handover steps."""
import httpx
import pytest
from test_api import chat
from test_booking_gate import GateModel

from restaurant import config, decisions
from restaurant.decide import Calibration, Decider

CALLBACK = {"kind": "other", "customer_name": "Ivan Petrov", "contact": "0888 123 456",
            "details": "Wants to talk to the staff about a booking"}


@pytest.fixture()
def client(app_url):
    with httpx.Client(base_url=app_url, timeout=30) as c:
        yield c


@pytest.fixture
def gate(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DECISION_GATE", True)
    question = decisions.WANTS_HUMAN
    Calibration(question.id, config.MODEL_NAME, question.prompt_hash(), 1.0, 30, "2026-11-24", {}, {}).save(tmp_path)
    model = GateModel()
    monkeypatch.setattr(decisions, "decider", lambda model_name=None: Decider(
        config.LLM_BASE_URL, model_name or config.MODEL_NAME, calibration_dir=tmp_path,
        transport=httpx.MockTransport(model.handle)))
    return model


def test_a_missed_request_gets_one_more_call_with_the_handover_step(client, fake_lm, gate):
    gate.humans.append({"Y": 0.96, "N": 0.04})
    fake_lm.script(reply="We have great pasta today!")
    fake_lm.script(reply="Of course. The restaurant staff can call you back: may I have your name and phone?")

    data = chat(client, "I'd like to speak to a real person, please.")

    assert data["reply"].startswith("Of course. The restaurant staff can call you back")
    assert len(fake_lm.requests) == 2
    first, second = (r["messages"][0]["content"] for r in fake_lm.requests)
    assert not first.endswith(decisions.HANDOVER_HINT) and second.endswith(decisions.HANDOVER_HINT)
    assert [m["role"] for m in fake_lm.requests[1]["messages"][1:]] == ["user"]   # the dropped answer is not kept


def test_no_extra_call_when_the_agent_passed_the_callback_on(client, fake_lm, gate):
    gate.humans.append({"Y": 0.99, "N": 0.01})
    fake_lm.script(("request_callback", CALLBACK), reply="Done. The restaurant staff will contact you.")
    data = chat(client, "Yes, please have someone call me.")
    assert data["reply"] == "Done. The restaurant staff will contact you."
    assert len(fake_lm.requests) == 2          # the tool round and the answer, no third call


@pytest.mark.parametrize("answer, calls", [({"Y": 0.5, "N": 0.5}, 2), ({"Y": 0.49, "N": 0.51}, 1), (500, 1)])
def test_threshold_and_no_usable_answer(client, fake_lm, gate, answer, calls):
    gate.humans.append(answer)
    fake_lm.script(reply="Which city?")
    fake_lm.script(reply="The staff can call you back.")
    chat(client, "hm, maybe a person?")
    assert len(fake_lm.requests) == calls


def test_the_question_sees_the_assistants_previous_message(client, fake_lm, gate):
    fake_lm.script(reply="Shall the restaurant staff call you back?")
    chat(client, "This is too complicated.")
    gate.humans.append({"Y": 0.9, "N": 0.1})
    fake_lm.script(reply="Sure.")
    fake_lm.script(reply="May I have your name and phone number?")
    chat(client, "yes")
    first, second = (r["messages"][1]["content"] for r in gate.human_requests)
    assert "(none: this is the guest's first message)" in first
    assert "The assistant's previous message: Shall the restaurant staff call you back?" in second
    assert second.endswith("<<input>>\nyes\n<</input>>")


def test_not_asked_when_switched_off(client, fake_lm, gate, monkeypatch):
    monkeypatch.setattr(config, "DECISION_GATE", False)
    fake_lm.script(reply="Hi.")
    chat(client, "I want a person.")
    assert gate.human_requests == [] and len(fake_lm.requests) == 1


def test_previous_assistant_message_is_shortened():
    assert decisions.previous_assistant_message([{"role": "assistant", "content": "x " * 300}]).endswith("...")
    assert decisions.previous_assistant_message([]) == "(none: this is the guest's first message)"
