import json
import math
from pathlib import Path

import pytest

from castiron import compile
from castiron.dag import Dag


def _cube_dag(size=20.0):
    """A minimal hand-written PartRoot DAG: a `size` cube via extrusion."""
    return {
        "version": "2.0",
        "rootNodeId": "part",
        "serializableNodes": {
            "FloatParameter": 8, "BoolParameter": 10, "StringParameter": 11,
            "Point": 4, "Line": 7, "Polygon": 18, "Sketch": 3, "Plane": 5,
            "Frame": 6, "Extrusion": 2, "PartRoot": 13,
        },
        "DAG": {
            "z": {"type": 8, "params": {"value": 0.0}, "deps": {}},
            "s": {"type": 8, "params": {"value": size}, "deps": {}},
            "no": {"type": 10, "params": {"value": False}, "deps": {}},
            "pn": {"type": 11, "params": {"value": "cube"}, "deps": {}},
            "frm": {"type": 6, "params": {"position": [0, 0, 0], "quaternion": [1, 0, 0, 0]}, "deps": {}},
            "pl": {"type": 5, "params": {}, "deps": {"frame": "frm"}},
            "a": {"type": 4, "params": {}, "deps": {"x": "z", "y": "z"}},
            "b": {"type": 4, "params": {}, "deps": {"x": "s", "y": "z"}},
            "c": {"type": 4, "params": {}, "deps": {"x": "s", "y": "s"}},
            "d": {"type": 4, "params": {}, "deps": {"x": "z", "y": "s"}},
            "l1": {"type": 7, "params": {}, "deps": {"p1": "a", "p2": "b"}},
            "l2": {"type": 7, "params": {}, "deps": {"p1": "b", "p2": "c"}},
            "l3": {"type": 7, "params": {}, "deps": {"p1": "c", "p2": "d"}},
            "l4": {"type": 7, "params": {}, "deps": {"p1": "d", "p2": "a"}},
            "poly": {"type": 18, "params": {}, "deps": {"lines": ["l1", "l2", "l3", "l4"]}},
            "sk": {"type": 3, "params": {}, "deps": {"plane": "pl", "elements": ["a", "b", "c", "d"]}},
            "ex": {"type": 2, "params": {}, "deps": {
                "shape": ["poly"], "sketch": "sk", "start": "z", "end": "s", "cut": "no"}},
            "part": {"type": 13, "params": {}, "deps": {
                "operations": ["ex"], "planes": ["pl"], "frame": "frm", "name": "pn"}},
        },
    }


def test_dag_tool_topo_order():
    dag = Dag.from_compiler_input(_cube_dag())
    order = dag.topo_order()
    # dependencies must appear before dependents
    assert order.index("z") < order.index("a")
    assert order.index("poly") < order.index("ex") < order.index("part")
    assert order[-1] == "part"


def test_compile_cube_stl_volume(tmp_path):
    m = compile(_cube_dag(20.0), format="stl", out_dir=str(tmp_path), name="cube")
    assert m.root_type == "PartRoot"
    assert len(m.parts) == 1
    assert m.parts[0]["volume"] == pytest.approx(8000.0)
    stl = Path(m.files[0])
    assert stl.exists() and stl.read_bytes()[:0] == b""  # non-empty binary
    assert stl.stat().st_size > 80
    # manifest written alongside
    assert (stl.parent / "manifest.json").exists()


def test_content_addressed_paths_are_stable(tmp_path):
    a = compile(_cube_dag(20.0), out_dir=str(tmp_path))
    b = compile(_cube_dag(20.0), out_dir=str(tmp_path))
    c = compile(_cube_dag(30.0), out_dir=str(tmp_path))
    assert a.dag_hash == b.dag_hash          # same DAG -> same hash/path
    assert a.dag_hash != c.dag_hash          # different DAG -> different path


def test_compile_from_foundation_part(tmp_path):
    foundation = pytest.importorskip("cadbuildr.foundation")
    from cadbuildr.foundation import Part, Sketch, Point, Line, Polygon, Extrusion

    class Cube(Part):
        def __init__(self, size=12):
            s = Sketch(self.xy())
            p1, p2, p3, p4 = Point(s, 0, 0), Point(s, size, 0), Point(s, size, size), Point(s, 0, size)
            poly = Polygon([Line(p1, p2), Line(p2, p3), Line(p3, p4), Line(p4, p1)])
            self.add_operation(Extrusion(poly, size))

    m = compile(Cube(12), format="stl", out_dir=str(tmp_path))
    assert m.parts[0]["volume"] == pytest.approx(12 ** 3)


_FIXTURES = Path(
    "/Users/clementjambou/src/cadbuildr/monorepo/tsjs/packages/cad/kernel-truck/src/__tests__/fixtures"
)


@pytest.mark.parametrize("fixture", [
    "cube", "cylinder", "chess_pawn", "donut",     # extrude / lathe
    "loft_basic", "sweep_spline",                   # loft / sweep (solid_from_mesh)
    "sm_l_bracket", "sm_base",                       # sheet metal
    "assy_pyramids",                                 # assembly (multi-part)
])
def test_fixture_corpus(fixture, tmp_path):
    fp = _FIXTURES / f"{fixture}.json"
    if not fp.exists():
        pytest.skip("kernel-truck fixture corpus not available")
    m = compile(json.loads(fp.read_text()), format="stl", out_dir=str(tmp_path))
    assert m.parts, f"{fixture} produced no parts"
    assert all(p["volume"] > 0 for p in m.parts), f"{fixture} has non-positive volume"
    assert all(Path(p["file"]).stat().st_size > 80 for p in m.parts)


def test_json_format(tmp_path):
    m = compile(_cube_dag(10.0), format="json", out_dir=str(tmp_path))
    data = json.loads(Path(m.files[0]).read_text())
    assert data["vertices"] and data["triangles"]
    assert data["volume"] == pytest.approx(1000.0)
