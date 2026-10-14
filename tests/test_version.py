"""The page version in web/version.js and web/sw.js must agree (both are bumped on every web change)."""
import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"


def test_version_matches_service_worker():
    v = re.search(r'APP_VERSION = "(v\d+)"', (WEB / "version.js").read_text(encoding="utf-8")).group(1)
    sw = re.search(r'VERSION = "(v\d+)"', (WEB / "sw.js").read_text(encoding="utf-8")).group(1)
    assert v == sw, f"web/version.js says {v} but web/sw.js says {sw}"
