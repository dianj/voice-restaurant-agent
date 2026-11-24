"""Grade the "wants a person" gate on a collected run: would the code start the callback steps (p(yes) >= 0.5,
calibrated as the app does)? A missed request (expected yes, no step) is the costly error; a needless one only offers
a callback.
    python evals/decisions/grade_wants_human.py [--suffix .qwen3.8-27b]
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from restaurant import decisions  # noqa: E402
from restaurant.decide import Calibration, apply_temperature  # noqa: E402

RUNS = ROOT / "evals" / "decisions" / "runs"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="", help="run file suffix, e.g. .qwen3.8-27b")
    parser.add_argument("--calibration", type=Path, default=ROOT / "evals" / "decisions" / "calibration")
    args = parser.parse_args(argv)
    question = decisions.WANTS_HUMAN
    run = json.loads((RUNS / f"{question.id}{args.suffix}.json").read_text(encoding="utf-8"))
    model = run["model"]
    cal = Calibration.load(args.calibration / model, question.id) or Calibration.load(args.calibration, question.id)
    usable = cal is not None and cal.model == model and cal.prompt_hash == question.prompt_hash()
    counts, wrong = Counter(), []
    for row in run["rows"]:
        probs = apply_temperature(row["probs_raw"], cal.temperature if usable else 1.0) if row["probs_raw"] else {}
        handover = bool(probs) and probs["yes"] >= decisions.HANDOVER_AT
        group = row["id"].split("-")[0]
        ok = handover == (row["expected"] == "yes")
        counts[(group, ok)] += 1
        if not ok:
            kind = "missed request" if row["expected"] == "yes" else "needless callback offer"
            wrong.append(f"{row['id']}: {kind} (p_yes {probs.get('yes', 0):.3f}) {row['text']!r}")
    latencies = [r["latency_ms"] for r in run["rows"]]
    print(f"model {model}, calibration {'T = ' + str(cal.temperature) if usable else 'none (T = 1)'}")
    print("| group | correct |")
    print("|---|---|")
    for group in dict.fromkeys(r["id"].split("-")[0] for r in run["rows"]):
        print(f"| {group} | {counts[(group, True)]}/{counts[(group, True)] + counts[(group, False)]} |")
    total = sum(counts[k] for k in counts if k[1])
    print(f"total {total}/{len(run['rows'])}; latency p50 {statistics.median(latencies):.0f} ms")
    print("\n".join(wrong) or "no errors")


if __name__ == "__main__":
    main()
