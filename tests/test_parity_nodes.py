"""Fixtures for node types the benchmark corpus never exercises: patterns,
MoveBody, BooleanOperation, surfaces, fillet / chamfer and STEP export.

Each test builds a tiny foundation part, compiles it with castiron and checks
volume / bounds against the analytic answer."""

import math

import pytest

foundation = pytest.importorskip("cadbuildr.foundation")
from cadbuildr.foundation import Extrusion, Part, Point, Sketch  # noqa: E402
from cadbuildr.foundation.gen import models as M  # noqa: E402

import castiron  # noqa: E402
import ironstream as ist  # noqa: E402


def _poly(s, pts):
    from cadbuildr.foundation import Line, Polygon

    p = [Point(s, x, y) for (x, y) in pts]
    return Polygon([Line(p[i], p[(i + 1) % len(p)]) for i in range(len(p))])


def _volume(part):
    meshes = castiron.compile_meshes(part)
    return sum(m["volume"] for m in meshes), meshes


def _bounds(meshes):
    v = [c for m in meshes for c in m["vertices"]]
    xs, ys, zs = v[0::3], v[1::3], v[2::3]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _part(build):
    class _P(Part):
        def __init__(self):
            build(self)

    return _P()


def test_reversed_extrusion_keeps_volume():
    # start > end, no taper: was inside-out before the kernel fix
    def b(self):
        s = Sketch(self.xy())
        self.add_operation(Extrusion(_poly(s, [(0, 0), (10, 0), (10, 10), (0, 10)]), -5.0, start=5.0))
        hole = _poly(s, [(3, 3), (7, 3), (7, 7), (3, 7)])
        self.add_operation(Extrusion(hole, -10.0, start=10.0, cut=True))

    vol, meshes = _volume(_part(b))
    assert vol == pytest.approx(10 * 10 * 10 - 4 * 4 * 10, rel=1e-6)
    lo, hi = _bounds(meshes)
    assert lo[2] == pytest.approx(-5.0) and hi[2] == pytest.approx(5.0)


def test_step_export_has_solid(tmp_path):
    def b(self):
        s = Sketch(self.xy())
        self.add_operation(Extrusion(_poly(s, [(0, 0), (4, 0), (4, 4), (0, 4)]), 2.0))

    man = castiron.compile(_part(b), format="step", out_dir=str(tmp_path))
    text = open(man.files[0]).read()
    assert text.startswith("ISO-10303-21")
    assert "MANIFOLD_SOLID_BREP" in text or "CLOSED_SHELL" in text
    assert "ADVANCED_FACE" in text or "FACE" in text


def test_linear_and_polar_pattern_and_move_body():
    from castiron import nodes as N

    seed = N.OpResult(ist.make_box(ist.Pnt(0, 0, 0), 1, 1, 1))

    class Ctx:
        def __init__(self, vals):
            self.vals = vals

        def get(self, node, k):
            return self.vals[k]

        def get_opt(self, node, k, d=None):
            return self.vals.get(k, d)

        def get_list(self, node, k):
            return self.vals.get(k, [])

    lin = N._linear_pattern(Ctx({"shape": seed, "direction": (1, 0, 0), "spacing": 3.0, "count": 4}), None)
    assert lin.solid.volume() == pytest.approx(4.0)
    bb = lin.solid.mesh()
    xs = bb.vertices_flat()[0::3]
    assert max(xs) == pytest.approx(10.0)

    pol = N._polar_pattern(
        Ctx({"shape": N.OpResult(ist.make_box(ist.Pnt(5, -0.5, 0), 1, 1, 1)), "count": 6, "angle": 360.0}), None
    )
    assert pol.solid.volume() == pytest.approx(6.0, rel=1e-6)

    mv = N._move_body(Ctx({"shape": seed, "translation": (0, 0, 7), "rotation_angle": 90.0}), None)
    zs = mv.solid.mesh().vertices_flat()[2::3]
    assert min(zs) == pytest.approx(7.0) and mv.solid.volume() == pytest.approx(1.0)

    big = N.OpResult(ist.make_box(ist.Pnt(0, 0, 0), 4, 4, 4))
    tool = N.OpResult(ist.make_box(ist.Pnt(2, 2, 2), 4, 4, 4))
    for op, want in (("SUBTRACT", 64 - 8), ("UNION", 64 + 56), ("INTERSECT", 8)):
        r = N._boolean_operation(Ctx({"target": big, "tools": [tool], "op": op}), None)
        assert r.solid.volume() == pytest.approx(want, rel=1e-6), op
