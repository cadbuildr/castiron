import json
import math
import os
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


# Optional DAG fixture corpus (CADbuildr monorepo, kernel-truck test fixtures).
# Point CASTIRON_FIXTURES_DIR at it to enable the corpus-backed tests.
_FIXTURES = Path(os.environ.get("CASTIRON_FIXTURES_DIR", "/nonexistent"))


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


def test_lathe_profile_straddling_axis():
    """A profile centred on the revolution axis (e.g. an ellipse) must be
    clipped to one side, else a 360° revolution double-covers it and the volume
    cancels to ~0. Regression for the ellipsoid example."""
    pytest.importorskip("cadbuildr.foundation")
    import math

    from cadbuildr.foundation import Axis, Ellipse, Line, Part, Point, Sketch
    from cadbuildr.foundation.gen.models.lathe import Lathe

    class Ellipsoid(Part):
        def __init__(self, a=10, b=30):
            s = Sketch(self.xy())
            ell = Ellipse(s.origin, a, b)          # semi-axes a (x), b (y)
            axis = Axis(Line(Point(s, 0, 0), Point(s, 1, 0)))  # X axis through centre
            self.add_operation(Lathe(ell, axis))

    m = compile(Ellipsoid(10, 30), format="stl", out_dir="/tmp/castiron_ellipsoid_test")
    # analytic ellipsoid volume 4/3 pi a b b = 4/3 pi 10 30 30 ~= 37699
    assert m.parts[0]["volume"] == pytest.approx(4 / 3 * math.pi * 10 * 30 * 30, rel=0.05)


def test_node_complete_against_foundation():
    """Every node type foundation can serialize has a castiron handler.

    The universe is the class set of cadbuildr.foundation.gen.models — the
    generated schema models. New foundation node types fail this test loudly
    until a handler (or explicit enum/ignore classification) exists.
    """
    import inspect

    models = pytest.importorskip("cadbuildr.foundation.gen.models")
    from castiron.nodes import HANDLERS

    names = {
        obj.__name__
        for n in dir(models)
        if inspect.isclass(obj := getattr(models, n))
        and obj.__module__.startswith("cadbuildr.foundation.gen.models")
    }
    missing = sorted(names - set(HANDLERS))
    assert not missing, f"foundation node types without castiron handler: {missing}"


@pytest.mark.parametrize("fixture", ["loft_basic", "sweep_spline", "multi_section_sweep", "sweep_twist"])
def test_loft_sweep_winding_outward(fixture, tmp_path):
    """Loft/sweep meshes must be watertight with globally-outward winding:
    trimesh's signed volume is positive and equals ironstream's."""
    trimesh = pytest.importorskip("trimesh")
    fp = _FIXTURES / f"{fixture}.json"
    if not fp.exists():
        pytest.skip("kernel-truck fixture corpus not available")
    m = compile(json.loads(fp.read_text()), format="stl", out_dir=str(tmp_path))
    t = trimesh.load(m.files[0])
    assert t.is_watertight and t.is_winding_consistent
    assert t.volume > 0
    assert t.volume == pytest.approx(m.parts[0]["volume"], rel=1e-6)


def test_svg_shape_loops_and_placement():
    """SVG paths/text flatten to loops; placement matches the reference kernel
    (scale + y-flip, bbox-centered, then shifted)."""
    from castiron.dag import DagNode
    from castiron.nodes import Loops, _svg_shape, classify_loops

    class Ctx:
        def __init__(self, values):
            self.values = values
        def get(self, node, key):
            return self.values[key]
        def get_opt(self, node, key, default=None):
            return self.values.get(key, default)

    svg = '''<svg viewBox="0 0 100 100">
      <path d="M10 10 L90 10 L90 90 L10 90 Z M40 40 L60 40 L60 60 L40 60 Z"/>
    </svg>'''
    ctx = Ctx({"svg": svg, "scale": 2.0, "angle": 0.0, "xshift": 5.0, "yshift": -3.0})
    node = DagNode(id="x", type_name="SVGShape")
    loops = _svg_shape(ctx, node)
    assert isinstance(loops, Loops) and len(loops.loops) == 2
    # bbox of the outer square is centered before shifting: center == (xshift, yshift)
    xs = [x for loop in loops.loops for x, _ in loop]
    ys = [y for loop in loops.loops for _, y in loop]
    assert (min(xs) + max(xs)) / 2 == pytest.approx(5.0)
    assert (min(ys) + max(ys)) / 2 == pytest.approx(-3.0)
    # inner square classified as a hole of the outer
    regions = classify_loops(loops.loops)
    assert len(regions) == 1 and len(regions[0][1]) == 1


def test_svg_text_renders_glyphs():
    pytest.importorskip("matplotlib")
    from castiron.nodes import _svg_text_loops

    loops = _svg_text_loops('<svg><text x="0" y="10" font-size="12">AB</text></svg>')
    # 'A' and 'B' both have holes -> at least 5 loops for two glyphs
    assert len(loops) >= 5


def test_tapered_extrusion_is_a_frustum(tmp_path):
    """A tapered circular extrusion must draft to a cone/frustum, not stay a
    straight cylinder. foundation's Cone compiles to Extrusion(circle, taper);
    endFactor = 1 - taper, so taper 0.8 takes r=10 down to r=2."""
    pytest.importorskip("cadbuildr.foundation")
    from cadbuildr.foundation import Circle, Part, Sketch
    from cadbuildr.foundation.gen.models.extrusion import Extrusion

    class Frustum(Part):
        def __init__(self):
            s = Sketch(self.xy())
            circle = Circle(s.origin, 10)
            self.add_operation(Extrusion(circle, 15, taper=0.8))

    m = compile(Frustum(), format="stl", out_dir=str(tmp_path))
    h, r1, r2 = 15.0, 10.0, 2.0
    frustum = math.pi * h / 3 * (r1 * r1 + r1 * r2 + r2 * r2)
    assert m.parts[0]["volume"] == pytest.approx(frustum, rel=2e-3)
    # a straight cylinder would be pi*100*15 = 4712 — guard against regressing.
    assert m.parts[0]["volume"] < 0.6 * math.pi * 100 * 15


def test_dxf_export_sketch(tmp_path):
    """A 2D sketch (square + circle) exports to a DXF whose entities are a real
    CIRCLE plus a closed polyline — the shapes the user drew, not their
    constituent edge primitives."""
    pytest.importorskip("cadbuildr.foundation")
    from cadbuildr.foundation import Circle, Part, Sketch, Square

    comp = Part()
    s = Sketch(comp.xy())
    Square.from_center_and_side(s.origin, 30)
    Circle(s.origin, 15)

    m = compile(s, format="dxf", out_dir=str(tmp_path), name="cs")
    txt = Path(m.files[0]).read_text()
    assert m.parts[0]["entities"] == 2          # one square loop + one circle
    assert txt.count("\nCIRCLE\n") == 1
    assert txt.count("\nLWPOLYLINE\n") == 1     # square, not its 4 edges
    assert "40\n15.000000" in txt               # circle radius preserved
    assert txt.rstrip().endswith("EOF")


def test_dxf_rejects_solid(tmp_path):
    """DXF is for 2D drawings; a solid design should steer the caller to STL/STEP."""
    from castiron.compiler import NothingToBuildError

    with pytest.raises(NothingToBuildError, match="2D sketch"):
        compile(_cube_dag(10.0), format="dxf", out_dir=str(tmp_path))


def test_json_format(tmp_path):
    m = compile(_cube_dag(10.0), format="json", out_dir=str(tmp_path))
    data = json.loads(Path(m.files[0]).read_text())
    assert data["vertices"] and data["triangles"]
    assert data["volume"] == pytest.approx(1000.0)
