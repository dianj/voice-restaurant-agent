"""How good the probabilities are: accuracy, expected calibration error, Brier score, risk-coverage, latency.

Each item is (probabilities per label, set of correct labels). The chosen label is the most probable one.
"""
import statistics


def chosen(probs: dict[str, float]) -> str:
    return max(probs, key=probs.get)


def accuracy(items) -> float:
    return sum(chosen(p) in ok for p, ok in items) / len(items) if items else 0.0


def ece(items, bins: int = 10) -> float:
    """Expected calibration error: |confidence - accuracy| per confidence bin, weighted by the bin's size."""
    if not items:
        return 0.0
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for probs, ok in items:
        label = chosen(probs)
        conf = probs[label]
        buckets[min(int(conf * bins), bins - 1)].append((conf, label in ok))
    total = len(items)
    return sum(len(b) / total * abs(statistics.fmean(c for c, _ in b) - statistics.fmean(int(h) for _, h in b))
               for b in buckets if b)


def brier(items) -> float:
    """Mean over cases of sum over labels (p - truth)^2; the correct mass counts as one target."""
    if not items:
        return 0.0
    total = 0.0
    for probs, ok in items:
        p_ok = sum(probs[label] for label in ok if label in probs)
        total += (1 - p_ok) ** 2 + sum(p ** 2 for label, p in probs.items() if label not in ok)
    return total / len(items)


def risk_coverage(items, thresholds=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)) -> list[dict]:
    """For each threshold: share of cases decided automatically (p >= t) and the accuracy among them."""
    rows = []
    for t in thresholds:
        kept = [(p, ok) for p, ok in items if p[chosen(p)] >= t]
        rows.append({"threshold": t, "coverage": len(kept) / len(items) if items else 0.0,
                     "accuracy": accuracy(kept) if kept else None, "n": len(kept)})
    return rows


def reliability(items, bins: int = 10) -> list[dict]:
    rows = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        inside = [(p[chosen(p)], chosen(p) in ok) for p, ok in items
                  if lo <= p[chosen(p)] < hi or (i == bins - 1 and p[chosen(p)] == 1.0)]
        if inside:
            rows.append({"lo": lo, "hi": hi, "n": len(inside), "confidence": statistics.fmean(c for c, _ in inside),
                         "accuracy": statistics.fmean(int(h) for _, h in inside)})
    return rows


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def summary(items) -> dict:
    return {"n": len(items), "accuracy": round(accuracy(items), 4), "ece": round(ece(items), 4),
            "brier": round(brier(items), 4)}
