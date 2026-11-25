"""Shared fixtures: a fake LM Studio, a fresh demo database per test, a frozen clock and the real app.

Environment variables must be set before `server` (and the `restaurant` package) is imported,
because they are read at import time.
"""
import os
import socket
import sys
import tempfile
import threading
import time
from datetime import datetime

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fake_lmstudio import FakeLMStudio  # noqa: E402

_TMP_DIR = tempfile.mkdtemp(prefix="restaurant-tests-")
_FAKE = FakeLMStudio().start()
os.environ["RESTAURANT_DB_PATH"] = os.path.join(_TMP_DIR, "test_restaurant.db")
os.environ["LLM_BASE_URL"] = _FAKE.base_url
os.environ["DECISION_GATE"] = "0"  # the booking gate has its own tests (tests/test_booking_gate.py)

import server  # noqa: E402  (must come after the environment is prepared)
from restaurant import clock, config, data, lmstudio  # noqa: E402

# Friday 20 November 2026, 10:00 in Sofia (08:00 UTC). Every test sees this as "now".
NOW = datetime(2026, 11, 20, 10, 0)

# One fresh database is built once and copied for each test (building it takes longer than copying).
_TEMPLATE_DB = os.path.join(_TMP_DIR, "template.db")
data.build_database(_TEMPLATE_DB)


@pytest.fixture(scope="session")
def fake_lm():
    yield _FAKE
    _FAKE.stop()


@pytest.fixture(autouse=True)
def clean_state(fake_lm, monkeypatch):
    """Every test starts from the demo data, no reservations, empty sessions, Gemma active and an empty script."""
    import shutil
    fake_lm.reset()
    fake_lm.loaded = {"gemma-4-26b-a4b-it"}
    lmstudio.model_state.update(active=config.MODEL_NAME, switching=False, target=None, error=None)
    shutil.copyfile(_TEMPLATE_DB, os.environ["RESTAURANT_DB_PATH"])
    server.sessions.clear()
    monkeypatch.setattr(clock, "utc_now", lambda: clock.sofia_to_utc(NOW))
    yield


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def app_url(fake_lm):
    """Run the real FastAPI app (uvicorn) in a background thread."""
    import uvicorn

    port = _free_port()
    uv = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not uv.started:
        if time.time() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    uv.should_exit = True
    thread.join(timeout=5)


def pytest_collection_modifyitems(items):
    """Browser tests last: Playwright's sync API leaves an event loop running in the main thread, after which
    asyncio.run() (the booking gate's tests) would fail."""
    items.sort(key=lambda item: "e2e" in item.nodeid.split("::")[0])
