"""OCCT parity + memory ceiling on real foundation DAGs.

Fixtures are cadbuildr-stdlib catalog parts (foundation 0.2.20 DAGs) the
kernel_compare benchmark flagged: thread sweeps were mirrored and their walls
twisted (DIN 934: IoU 0.30), and threaded rods OOM'd above 2.5 GB. Each must
match the OCCT kernel's mesh volume within 1 % in a child process whose peak
RSS stays under the ceiling."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

FIX = Path(__file__).parent / "fixtures" / "corpus"

# (fixture, OCCT volume mm^3, peak RSS ceiling MB)
CASES = [
    ("gt2-pulley.json", 992.353, 300),
    ("lead-screw-nut-t8.json", 1406.217, 300),
    ("din912-socket-cap-screw.json", 2512.474, 400),
    ("threaded-rod-metric.json", 5479.605, 600),
]

CHILD = r"""
import json, resource, sys
import castiron
meshes = castiron.compile_meshes(json.load(open(sys.argv[1])))
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_mb = peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
print(json.dumps({"volume": sum(m["volume"] for m in meshes), "peak_mb": peak_mb}))
"""


@pytest.mark.parametrize("name,occt_volume,ceiling_mb", CASES)
def test_corpus_part_matches_occt_under_ceiling(name, occt_volume, ceiling_mb):
    out = subprocess.run(
        [sys.executable, "-c", CHILD, str(FIX / name)], capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stderr[-2000:]
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["volume"] == pytest.approx(occt_volume, rel=0.01)
    assert r["peak_mb"] < ceiling_mb, f"{name}: peak {r['peak_mb']:.0f} MB"
