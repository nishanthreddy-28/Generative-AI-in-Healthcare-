"""No paid/external network calls in the test suite."""
import os
from pathlib import Path
import socket
import sys
import tempfile
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT))
TEMP = ROOT / ".runtime" / "test_temp"
TEMP.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(TEMP)
os.environ["TEMP"] = str(TEMP)
os.environ["TMP"] = str(TEMP)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("External network access is forbidden in automated tests.")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("APP_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("APP_PUBLIC_DEPLOYMENT", raising=False)
