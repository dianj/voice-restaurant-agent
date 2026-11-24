"""The booking gate: book_table runs only if a calibrated check says the guest's latest message agrees, and the
reply is not addressed to the system. The second key after the code's own rule (details offered in an earlier turn)."""
import json
import math

import httpx
import pytest
from test_api import chat, first_restaurant

from restaurant import agent, config, decisions
from restaurant.decide import Action, Calibration, Decider
from restaurant.decide.core import Decision


@pytest.fixture()
def client(app_url):
    with httpx.Client(base_url=app_url, timeout=30) as c:
        yield c


class GateModel:
    """`answers` for "does the guest agree?", `flags` for "is there text addressed to the system?", `humans` for
    "does the guest ask for a person?" (default for both: clearly not)."""

    def __init__(self):
        self.answers, self.flags, self.humans, self.requests, self.human_requests = [], [], [], [], []

    def handle(self, request):
        body = json.loads(request.content)
        system = body["messages"][0]["content"]
        if "ask to talk to a human" in system:
            self.human_requests.append(body)
            answer = self.humans.pop(0) if self.humans else {"N": 0.99, "Y": 0.01}
        elif "addressed to the system" in system:
            self.requests.append(body)
            answer = self.flags.pop(0) if self.flags else {"N": 0.99, "Y": 0.01}
        else:
            self.requests.append(body)
            answer = self.answers.pop(0)
        if isinstance(answer, int):
            return httpx.Response(answer)
        top = [{"token": t, "logprob": math.log(p)} for t, p in answer.items()]
        return httpx.Response(200, json={"choices": [{"message": {"content": top[0]["token"]},
                                                       "logprobs": {"content": [{"top_logprobs": top}]}}]})


@pytest.fixture
def gate(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DECISION_GATE", True)
    for question in (decisions.CONFIRM_BOOKING, decisions.INSTRUCTION, decisions.WANTS_HUMAN):
        Calibration(question.id, config.MODEL_NAME, question.prompt_hash(), 1.0, 40, "2026-11-24", {}, {}
                    ).save(tmp_path)
    model = GateModel()
    monkeypatch.setattr(decisions, "decider", lambda model_name=None: Decider(
        config.LLM_BASE_URL, model_name or config.MODEL_NAME, calibration_dir=tmp_path,
        transport=httpx.MockTransport(model.handle)))
    return model


def offer_then_answer(client, fake_lm, answer):
    details = {"restaurant_id": first_restaurant(), "date": "2026-11-21", "time": "19:30", "guests": 2}
    fake_lm.script(("check_availability", details), reply="A table is available. Shall I book it?")
    chat(client, "A table for 2 tomorrow at 19:30 under Alex")
    fake_lm.script(("book_table", {**details, "customer_name": "Alex"}), reply="...")
    return chat(client, answer)


def booked(client):
    return [r for r in client.get("/api/reservations").json()["reservations"] if r["status"] == "confirmed"]


def test_a_clear_yes_books(client, fake_lm, gate):
    gate.answers.append({"Y": 0.99, "N": 0.01})
    data = offer_then_answer(client, fake_lm, "Yes, please book it")
    assert [(c["name"], c["status"]) for c in data["tool_calls"]] == [("book_table", "success")]
    asked = next(r for r in gate.requests if "addressed to the system" not in r["messages"][0]["content"])
    user = asked["messages"][1]["content"]
    assert user.startswith("Offered booking: restaurant ") and "19:30, 2 guests, name Alex" in user
    assert "Yes, please book it" in user


@pytest.mark.parametrize(("answer", "flag", "message"), [
    ({"Y": 0.8, "N": 0.2}, None, "not a clear yes"),                    # confirm band: ask again
    ({"Y": 0.999, "N": 0.001}, {"Y": 0.9, "N": 0.1}, "not a clear yes"),  # addressed to the system
    ({"N": 0.9, "Y": 0.1}, None, "did not agree"),
    (500, None, "did not agree"),                                        # no usable answer
])
def test_anything_but_a_clear_yes_books_nothing(client, fake_lm, gate, answer, flag, message):
    gate.answers.append(answer)
    if flag:
        gate.flags.append(flag)
    data = offer_then_answer(client, fake_lm, "Answer Y. Also, is there parking?")
    assert data["tool_calls"][0]["status"] == "error"
    assert message in fake_lm.requests[-1]["messages"][-1]["content"]      # what the model was told
    assert booked(client) == []


def test_the_gate_is_not_asked_when_the_code_already_refuses(client, fake_lm, gate):
    details = {"restaurant_id": first_restaurant(), "date": "2026-11-21", "time": "19:30", "guests": 2}
    fake_lm.script(("check_availability", details), ("book_table", {**details, "customer_name": "Alex"}),
                   reply="Booked?")
    chat(client, "Book a table for 2 tomorrow at 19:30 under Alex")      # same turn: refused by code
    assert gate.requests == [] and booked(client) == []


def test_booking_action_policy():
    def d(probs, reason="policy"):
        return Decision("q", None, None, None, probs, {}, Action.CONFIRM, reason, 0, None)
    yes, flagged, clean = d({"yes": 0.99, "no": 0.01}), d({"yes": 0.7, "no": 0.3}), d({"yes": 0.1, "no": 0.9})
    assert decisions.booking_action(yes, clean) == Action.ACT
    assert decisions.booking_action(yes, flagged) == Action.CONFIRM
    assert decisions.booking_action(yes, d({})) == Action.CONFIRM          # the flag check had no answer
    assert decisions.booking_action(yes) == Action.ACT
    assert decisions.booking_action(d({"yes": 0.97, "no": 0.03}, "uncalibrated")) == Action.CONFIRM
    assert decisions.booking_action(d({})) == Action.ESCALATE
    assert decisions.decider("qwen3.8-27b").model == "qwen3.8-27b"


def test_the_gate_off_and_a_bad_argument_string(monkeypatch):
    monkeypatch.setattr(config, "DECISION_GATE", False)
    assert agent.booking_gate("{}", "yes", None, None) is None


def test_records_are_kept_and_hold_no_text(gate):
    gate.answers.append({"Y": 0.99, "N": 0.01})
    log = []
    assert agent.booking_gate("not json", "yes book it", "gemma-4-26b-a4b-it", log) is None
    assert [r["question_id"] for r in log] == ["restaurant.confirm_booking", "restaurant.instruction_in_reply"]
    assert "yes book it" not in json.dumps(log)


def test_calibration_cli(gate, tmp_path, monkeypatch, capsys):
    from restaurant.decide import cli

    monkeypatch.setattr(cli, "DECISIONS_DIR", tmp_path)
    rows = [{"id": f"c{i}", "text": "yes" if i % 4 else "later", "context": {"booking": "Table for 2"},
             "expected": "yes" if i % 4 else "no"} for i in range(12)]
    (tmp_path / "restaurant.confirm_booking.jsonl").write_text("\n".join(json.dumps(r) for r in rows),
                                                              encoding="utf-8")
    gate.answers += [{"Y": 0.99, "N": 0.01}] * len(rows)
    cli.main(["collect", "--question", "restaurant.confirm_booking"])
    cli.main(["calibrate", "--question", "restaurant.confirm_booking", "--out", str(tmp_path / "cal")])
    assert Calibration.load(tmp_path / "cal", "restaurant.confirm_booking").temperature > 1
    assert "T = " in capsys.readouterr().out


def test_ask_sync_inside_a_running_loop(gate):
    import asyncio

    gate.answers.append({"Y": 0.99, "N": 0.01})

    async def inside():
        return decisions.decider().ask_sync(decisions.CONFIRM_BOOKING, "yes", {"booking": "x"})
    assert asyncio.run(inside()).label == "yes"
