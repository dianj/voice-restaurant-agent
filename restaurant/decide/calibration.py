"""Temperature scaling: one number T per question, fitted on labelled cases (pure Python).

p_i ∝ exp(log p_i / T). T > 1 softens an over-confident model. T is found by minimising the negative log-likelihood
of the correct answers (golden-section search on log T), and it is never below 1: a gate is not allowed to become
more confident than the raw model. (A calibration set without errors would otherwise push T towards 0 and every
answer to 100 %, which says more about the set than about the model.) A case may accept
several labels (e.g. "Threat" or "Harassment"): its likelihood is the summed probability of those labels.
"""
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

FLOOR = 1e-9
T_MIN, T_MAX = 1.0, 50.0


def apply_temperature(probs: dict[str, float], temperature: float) -> dict[str, float]:
    if temperature == 1.0:
        return dict(probs)
    scaled = {k: math.log(max(p, FLOOR)) / temperature for k, p in probs.items()}
    top = max(scaled.values())
    exps = {k: math.exp(v - top) for k, v in scaled.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def nll(cases: list[tuple[dict[str, float], set[str]]], temperature: float) -> float:
    total = 0.0
    for probs, correct in cases:
        scaled = apply_temperature(probs, temperature)
        total -= math.log(max(sum(scaled.get(label, 0.0) for label in correct), FLOOR))
    return total / max(len(cases), 1)


def fit_temperature(cases: list[tuple[dict[str, float], set[str]]], iterations: int = 80) -> float:
    lo, hi = math.log(T_MIN), math.log(T_MAX)
    ratio = (math.sqrt(5) - 1) / 2
    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = nll(cases, math.exp(a)), nll(cases, math.exp(b))
    for _ in range(iterations):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = nll(cases, math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = nll(cases, math.exp(b))
    return round(math.exp((lo + hi) / 2), 4)


@dataclass
class Calibration:
    question_id: str
    model: str
    prompt_hash: str
    temperature: float
    n_cases: int
    fitted_at: str
    metrics_before: dict
    metrics_after: dict

    @staticmethod
    def path(directory: Path, question_id: str) -> Path:
        return Path(directory) / f"{question_id}.json"

    @classmethod
    def load(cls, directory: Path, question_id: str) -> "Calibration | None":
        path = cls.path(directory, question_id)
        if not path.exists():
            return None
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def save(self, directory: Path) -> Path:
        path = self.path(directory, self.question_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path
