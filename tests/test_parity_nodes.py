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


def _box_part(fillet=None, chamfer=None, finder=None, cyl=False):
    from cadbuildr.foundation import Circle

    def b(self):
        s = Sketch(self.xy())
        shape = Circle(s.origin, 5.0) if cyl else _poly(s, [(0, 0), (10, 0), (10, 10), (0, 10)])
        e = Extrusion(shape, 10.0)
        self.add_operation(e)
        if fillet:
            self.add_operation(M.Fillet(e, fillet, finder))
        if chamfer:
            self.add_operation(M.Chamfer(e, chamfer, finder))

    return _part(b)


def _finder(rule):
    return M.EdgeFinder(rule)


def test_fillet_vertical_box_edges():
    # 4 vertical edges, r=1: each removes (1 - pi/4) * 10
    vol, _ = _volume(_box_part(fillet=1.0, finder=_finder(M.ParallelToAxisRule("Z"))))
    assert vol == pytest.approx(1000 - 4 * (1 - math.pi / 4) * 10, rel=2e-3)


def test_chamfer_one_edge_by_box():
    # the vertical edge at (10, 10): a 1 x 1 right triangle along 10 mm
    rule = M.AndFinderRule([M.ParallelToAxisRule("Z"), M.InBoxFinderRule(
        M.Point3D(9.5, 9.5, -1.0), M.Point3D(10.5, 10.5, 11.0))])
    vol, meshes = _volume(_box_part(chamfer=1.0, finder=_finder(rule)))
    assert vol == pytest.approx(1000 - 0.5 * 10, rel=1e-3)


def test_fillet_all_box_edges_rounds_everything():
    vol, meshes = _volume(_box_part(fillet=1.0))
    # exact rounded box (Minkowski of an 8 mm cube and r=1): a^3 + 6a^2 r + 3 pi a r^2 + 4/3 pi r^3.
    # Corners here are edge roll-off intersections, not sphere octants: ~1 %.
    a, r = 8.0, 1.0
    exact = a**3 + 6 * a * a * r + 3 * math.pi * a * r * r + 4 / 3 * math.pi * r**3
    assert vol == pytest.approx(exact, rel=1e-2)
    lo, hi = _bounds(meshes)
    assert lo == pytest.approx((0, 0, 0), abs=1e-6) and hi == pytest.approx((10, 10, 10), abs=1e-6)


def test_fillet_cylinder_top_circle():
    # r=1 fillet on both circular edges of an r=5 cylinder:
    # removed area (1 - pi/4) revolved at its centroid radius
    a = 1 - math.pi / 4
    xc = (10 - 3 * math.pi) / (12 - 3 * math.pi)  # centroid offset from the corner
    removed = 2 * math.pi * (5 - xc) * a
    vol, _ = _volume(_box_part(fillet=1.0, cyl=True, finder=_finder(M.IsCircleRule())))
    full = math.pi * 25 * 10 * (64 * math.sin(2 * math.pi / 64) / (2 * math.pi))
    assert vol == pytest.approx(full - 2 * removed, rel=1e-2)
