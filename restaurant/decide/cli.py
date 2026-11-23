"""Collect the model's answers on labelled cases, fit the temperature, write the calibration file and a report.

    python -m restaurant.decide.cli collect --question restaurant.confirm_booking   # evals/decisions/<question>.jsonl
    python -m restaurant.decide.cli calibrate --question restaurant.confirm_booking

`collect` asks the configured model once per case and saves the raw probabilities (evals/decisions/runs/), so
`calibrate` can be repeated without the model. "After" metrics are 5-fold cross-validated: each fold is scored
with a temperature fitted on the other four, so the numbers are not measured on the data the fit saw.
Cases: one JSON object per line: {"id", "text", "context"?, "expected": label | [labels] | "unclear", "tags"}.
"""
import argparse
import asyncio
import json
import random
from datetime import UTC, datetime
from pathlib import Path

from . import metrics
from .calibration import Calibration, apply_temperature, fit_temperature

ROOT = Path(__file__).resolve().parents[2]
DECISIONS_DIR = ROOT / "evals" / "decisions"
FOLDS = 5


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def labelled(rows: list[dict]) -> list[tuple[dict, set]]:
    """(raw probabilities, correct labels) for the cases with a label and a usable answer."""
    out = []
    for row in rows:
        expected = row["expected"]
        if expected == "unclear" or not row.get("probs_raw"):
            continue
        out.append((row["probs_raw"], set(expected if isinstance(expected, list) else [expected])))
    return out


def cross_validated(items: list[tuple[dict, set]], folds: int = FOLDS, seed: int = 7) -> list[tuple[dict, set]]:
    """Folds over a seeded shuffle: case files are grouped by kind, so plain i % folds would put one kind in
    one fold."""
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    fold_of = {index: position % folds for position, index in enumerate(order)}
    scored = []
    for k in range(folds):
        train = [x for i, x in enumerate(items) if fold_of[i] != k]
        test = [x for i, x in enumerate(items) if fold_of[i] == k]
        t = fit_temperature(train) if train else 1.0
        scored += [(apply_temperature(p, t), ok) for p, ok in test]
    return scored


def reliability_svg(before: list[dict], after: list[dict], title: str) -> str:
    size, pad = 320, 40
    inner = size - 2 * pad

    def xy(conf, acc):
        return pad + conf * inner, size - pad - acc * inner

    def dots(rows, color):
        out = []
        for r in rows:
            x, y = xy(r["confidence"], r["accuracy"])
            radius = 3 + min(9, r["n"] ** 0.5)
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.65">'
                       f'<title>confidence {r["confidence"]:.2f}, accuracy {r["accuracy"]:.2f}, n={r["n"]}</title>'
                       f'</circle>')
        return "".join(out)
    x0, y0 = xy(0, 0)
    x1, y1 = xy(1, 1)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size + 30}" font-family="sans-serif" '
            f'font-size="11"><rect width="100%" height="100%" fill="#fff"/>'
            f'<text x="{pad}" y="18" font-size="13">{title}</text>'
            f'<rect x="{pad}" y="{pad}" width="{inner}" height="{inner}" fill="none" stroke="#999"/>'
            f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="#bbb" stroke-dasharray="4 3"/>'
            f'{dots(before, "#d9534f")}{dots(after, "#2a7ab9")}'
            f'<text x="{pad}" y="{size - 12}">confidence →</text>'
            f'<text x="12" y="{pad + 10}" transform="rotate(-90 12 {pad + 10})" text-anchor="end">accuracy →</text>'
            f'<circle cx="{pad + 10}" cy="{size + 14}" r="5" fill="#d9534f"/><text x="{pad + 20}" y="{size + 18}">'
            f'raw</text><circle cx="{pad + 70}" cy="{size + 14}" r="5" fill="#2a7ab9"/>'
            f'<text x="{pad + 80}" y="{size + 18}">calibrated (cross-validated)</text></svg>\n')


def report(question_id: str, run: dict, cal: Calibration, raw_items, cv_items, rows: list[dict]) -> str:
    def fmt(v):
        return "—" if v is None else f"{v:.3f}"
    lines = [f"# Calibration: {question_id}", "",
             f"Model `{run['model']}`, prompt `{run['prompt_hash']}`, {cal.n_cases} labelled cases, "
             f"fitted temperature **T = {cal.temperature}** ({cal.fitted_at[:10]}).", "",
             "| | accuracy | ECE | Brier |", "|---|---|---|---|",
             f"| raw (T = 1) | {fmt(cal.metrics_before['accuracy'])} | {fmt(cal.metrics_before['ece'])} | "
             f"{fmt(cal.metrics_before['brier'])} |",
             f"| calibrated, 5-fold cross-validated | {fmt(cal.metrics_after['accuracy'])} | "
             f"{fmt(cal.metrics_after['ece'])} | {fmt(cal.metrics_after['brier'])} |", "",
             "Risk-coverage (calibrated): the share of cases at or above each threshold and their accuracy.", "",
             "| threshold | coverage | accuracy | n |", "|---|---|---|---|"]
    for r in metrics.risk_coverage(cv_items):
        lines.append(f"| {r['threshold']} | {r['coverage']:.2f} | {fmt(r['accuracy'])} | {r['n']} |")
    unclear = [r for r in rows if r["expected"] == "unclear"]
    if unclear:
        lines += ["", f"Cases labelled *unclear* (graded by action, not label): {len(unclear)}."]
    lines += ["", f"![Reliability diagram]({question_id}.svg)", ""]
    return "\n".join(lines)


async def collect(question_id: str, cases_path: Path, out: Path) -> dict:
    from restaurant.decisions import QUESTIONS, decider  # the project's questions and model settings

    question = QUESTIONS[question_id]
    gate = decider()
    rows = []
    for case in load_cases(cases_path):
        probs, reason, ms = await gate.raw_probabilities(question, case["text"], case.get("context"))
        rows.append({**case, "probs_raw": probs, "reason": reason, "latency_ms": ms})
        print(f"{case['id']:<16} {reason:<22} {max(probs, key=probs.get) if probs else '-':<18} {ms} ms")
    run = {"question_id": question_id, "model": gate.model, "prompt_hash": question.prompt_hash(),
           "collected_at": datetime.now(UTC).isoformat(), "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(run, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return run


def calibrate(question_id: str, run_path: Path, out_dir: Path) -> Calibration:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    items = labelled(run["rows"])
    temperature = fit_temperature(items)
    cv_items = cross_validated(items)
    cal = Calibration(question_id=question_id, model=run["model"], prompt_hash=run["prompt_hash"],
                      temperature=temperature, n_cases=len(items), fitted_at=datetime.now(UTC).isoformat(),
                      metrics_before=metrics.summary(items), metrics_after=metrics.summary(cv_items))
    cal.save(out_dir)
    (out_dir / f"{question_id}.svg").write_text(
        reliability_svg(metrics.reliability(items), metrics.reliability(cv_items), question_id), encoding="utf-8")
    (out_dir / f"{question_id}.md").write_text(report(question_id, run, cal, items, cv_items, run["rows"]),
                                               encoding="utf-8")
    return cal


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m restaurant.decide.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--question", required=True)
    c.add_argument("--cases", type=Path)
    k = sub.add_parser("calibrate")
    k.add_argument("--question", required=True)
    k.add_argument("--run", type=Path)
    k.add_argument("--out", type=Path, default=DECISIONS_DIR / "calibration")
    args = parser.parse_args(argv)
    run_path = DECISIONS_DIR / "runs" / f"{args.question}.json"
    if args.command == "collect":
        asyncio.run(collect(args.question, args.cases or DECISIONS_DIR / f"{args.question}.jsonl", run_path))
        print(f"saved {run_path}")
    else:
        cal = calibrate(args.question, args.run or run_path, args.out)
        print(f"T = {cal.temperature}; before {cal.metrics_before}; after (cross-validated) {cal.metrics_after}")


if __name__ == "__main__":  # pragma: no cover
    main()
