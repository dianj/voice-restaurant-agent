"""What is kept of every decision: enough to audit and measure it, without the text itself (only its hash)."""
from datetime import datetime

from pydantic import BaseModel


class DecisionRecord(BaseModel):
    ts: datetime
    question_id: str
    prompt_version: str
    prompt_hash: str
    model: str
    decide_version: str
    labels: list[str]
    p_raw: dict[str, float]
    p: dict[str, float]
    chosen: str | None
    action: str
    reason: str               # policy | uncalibrated | unknown | low_format_confidence
    temperature: float | None
    latency_ms: int
    input_sha256: str
