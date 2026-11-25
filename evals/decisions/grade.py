"""Grade the confirmation gate as a whole on the collected runs: what would the policy do with each reply?

A "yes" reply must run the action (ACT); a "no" or "unclear" reply must not. Run after `collect` for both questions:
    python evals/decisions/grade.py [--suffix .qwen3.8-27b] [--question restaurant.confirm_step]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from restaurant import decisions  # noqa: E402
from restaurant.decide import Action, Calibration, apply_temperature  # noqa: E402
from restaurant.decide.core import Decision  # noqa: E402

RUNS = ROOT / "evals" / "decisions" / "runs"


def decision(question, row, cal_dir: Path, model: str) -> Decision:
    cal = Calibration.load(cal_dir / model, question.id) or Calibration.load(cal_dir, question.id)  # as the app
    usable = cal is not None and cal.model == model and cal.prompt_hash == question.prompt_hash()
    probs = apply_temperature(row["probs_raw"], cal.temperature if usable else 1.0) if row["probs_raw"] else {}
    label = max(probs, key=probs.get) if probs else None
    return Decision(question.id, label, probs.get(label), None, probs, row["probs_raw"] or {}, Action.CONFIRM,
                    "policy" if usable else "uncalibrated", row["latency_ms"], None)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="", help="run file suffix, e.g. .qwen3.8-27b")
    parser.add_argument("--question", default=decisions.CONFIRM_BOOKING.id,
                        help="the yes gate: restaurant.confirm_booking or restaurant.confirm_step")
    parser.add_argument("--calibration", type=Path, default=ROOT / "evals" / "decisions" / "calibration")
    args = parser.parse_args(argv)
    question = decisions.QUESTIONS[args.question]
    agree_run = json.loads((RUNS / f"{question.id}{args.suffix}.json").read_text(encoding="utf-8"))
    flag_run = json.loads((RUNS / f"restaurant.instruction_in_reply{args.suffix}.json").read_text(encoding="utf-8"))
    flags = {r["id"]: r for r in flag_run["rows"]}
    model = agree_run["model"]
    counts, wrong = Counter(), []
    for row in agree_run["rows"]:
        agree = decision(question, row, args.calibration, model)
        flag = decision(decisions.INSTRUCTION, flags[row["id"]], args.calibration, model)
        raw = Decision(agree.question_id, None, None, None, row["probs_raw"] or {}, {}, Action.CONFIRM, "policy",
                       0, None)
        alone = decisions.booking_action(raw, None, question)   # the first question alone, raw probabilities (T = 1)
        both = decisions.booking_action(agree, flag, question)
        group = row["id"].removeprefix("step-").split("-")[0]
        ok = (both == Action.ACT) == (row["expected"] == "yes")
        counts[(group, "ok" if ok else "wrong")] += 1
        counts[("acted_alone_wrongly", group)] += (alone == Action.ACT) and row["expected"] != "yes"
        if not ok:
            wrong.append(f"{row['id']}: expected {row['expected']}, policy {both.value} "
                         f"(p_yes {agree.probs.get('yes', 0):.3f}, p_flag {flag.probs.get('yes', 0):.3f})")
    groups = sorted({g for g, _ in counts if g != "acted_alone_wrongly"})
    print(f"model {model}")
    print("| group | correct | would have acted wrongly: first question alone, T = 1 |")
    print("|---|---|---|")
    for g in groups:
        total = counts[(g, "ok")] + counts[(g, "wrong")]
        print(f"| {g} | {counts[(g, 'ok')]}/{total} | {counts[('acted_alone_wrongly', g)]} |")
    print("\n".join(wrong) or "no errors")


if __name__ == "__main__":
    main()
