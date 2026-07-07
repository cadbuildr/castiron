"""Per-node conversion functions: foundation DAG node -> IronStream binding calls.

Each handler has the signature ``handler(ctx, node) -> value`` where ``ctx`` is
the :class:`~castiron.compiler.Compiler` (it exposes the resolved-dependency
store and helpers) and ``value`` is whatever downstream nodes consume:

* parameter nodes           -> the raw scalar (``float`` / ``int`` / ``str`` / ``bool``)
* ``Point``                 -> a 2D sketch-local ``(x, y)`` tuple
* segments (Line, Arc)      -> a **polyline**: a list of 2D points start..end
* closed shapes             -> a closed loop: a list of 2D points
* ``Frame`` / ``Plane`` / ``Sketch`` -> a :class:`~castiron.placement.Placement`
* ``Axis``                  -> ``("axis", location3d, direction3d)``
* operations (Extrusion, Lathe, ...) -> an :class:`OpResult`
* ``Part`` / ``PartRoot``   -> a :class:`PartResult`

New node types are added by writing a handler and registering it in ``HANDLERS``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import ironstream as ist

from .loft import _cross, _norm, loft_solid, sweep_solid
from .placement import Placement


@dataclass
class OpResult:
    """A solid produced by one operation, plus whether it is subtractive."""

    solid: "ist.Solid"
    cut: bool = False


@dataclass
class PartResult:
    name: str
    solid: "ist.Solid"


# Sentinel returned by metadata-only nodes (e.g. Material) that produce no geometry.
METADATA = object()


class UnsupportedNodeError(NotImplementedError):
    pass


# --- helpers ---------------------------------------------------------------


def _vscale(v, s):
    return (v[0] * s, v[1] * s, v[2] * s)


def _shapes(ctx, node, key="shape"):
    """Resolve a ``shape`` dep that may be a single ref or a list of refs."""
    ref = node.deps[key]
    if isinstance(ref, (list, tuple)):
        return [ctx.store[r] for r in ref]
    return [ctx.store[ref]]


def _loop_to_wire(placement: Placement, loop, z=0.0):
    pts = [ist.Pnt(*placement.to_world(x, y, z)) for (x, y) in loop]
    return ist.make_polygon(pts)


def _loop_from_segments(ctx, node):
    """Concatenate segment polylines into one loop (drop each shared endpoint)."""
    for key in ("primitives", "lines", "elements"):
        if key in node.deps:
            segs = ctx.get_list(node, key)
            break
    else:
        return []
    pts: list[tuple[float, float]] = []
    for seg in segs:
        pts.extend(seg[:-1] if len(seg) > 1 else seg)
    return pts


# --- parameters ------------------------------------------------------------


def _param(ctx, node):
    return node.params.get("value")


def _material(ctx, node):
    return METADATA


# --- sketch primitives (points / segments) --------------------------------


def _point(ctx, node):
    return (float(ctx.get(node, "x")), float(ctx.get(node, "y")))


def _line(ctx, node):
    return [ctx.get(node, "p1"), ctx.get(node, "p2")]


def _circumcenter(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return (ux, uy)


def _arc(ctx, node, segments=24):
    """3-point arc (start ``p1``, through ``p2``, end ``p3``) -> polyline."""
    p1, p2, p3 = ctx.get(node, "p1"), ctx.get(node, "p2"), ctx.get(node, "p3")
    c = _circumcenter(p1, p2, p3)
    if c is None:
        return [p1, p3]
    r = math.hypot(p1[0] - c[0], p1[1] - c[1])
    ang = lambda p: math.atan2(p[1] - c[1], p[0] - c[0])
    ccw = lambda x: x % (2 * math.pi)
    a1 = ang(p1)
    d13, d12 = ccw(ang(p3) - a1), ccw(ang(p2) - a1)
    sweep = d13 if d12 <= d13 else d13 - 2 * math.pi  # pick the arc passing p2
    return [
        (c[0] + r * math.cos(a1 + sweep * i / segments),
         c[1] + r * math.sin(a1 + sweep * i / segments))
        for i in range(segments + 1)
    ]


# --- closed shapes (loops) -------------------------------------------------


def _polygon(ctx, node):
    return _loop_from_segments(ctx, node)


def _circle(ctx, node, segments=64):
    cx, cy = ctx.get(node, "center")
    r = float(ctx.get(node, "radius"))
    return [(cx + r * math.cos(2 * math.pi * i / segments),
             cy + r * math.sin(2 * math.pi * i / segments)) for i in range(segments)]


def _ellipse(ctx, node, segments=64):
    a = float(ctx.get(node, "a"))
    b = float(ctx.get(node, "b"))
    cx, cy = ctx.get(node, "center")
    return [(cx + a * math.cos(2 * math.pi * i / segments),
             cy + b * math.sin(2 * math.pi * i / segments)) for i in range(segments)]


# --- construction geometry -------------------------------------------------


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _compose(parent: Placement, local: Placement) -> Placement:
    return Placement(parent.to_world(*local.position),
                     _quat_mul(parent.quaternion, local.quaternion))


def _frame(ctx, node):
    pos = tuple(node.params.get("position", (0.0, 0.0, 0.0)))
    quat = tuple(node.params.get("quaternion", (1.0, 0.0, 0.0, 0.0)))
    local = Placement(pos, quat)
    parent = ctx.get_opt(node, "top_frame")
    return local if parent is None else _compose(parent, local)


def _plane(ctx, node):
    return ctx.get(node, "frame")


def _sketch(ctx, node):
    return ctx.get(node, "plane")


def _axis(ctx, node):
    placement: Placement = ctx.get(node, "sketch")
    line = ctx.get(node, "line")  # polyline [p1, p2]
    w1 = placement.to_world(*line[0], 0.0)
    w2 = placement.to_world(*line[-1], 0.0)
    direction = (w2[0] - w1[0], w2[1] - w1[1], w2[2] - w1[2])
    return ("axis", w1, direction)


# --- operations ------------------------------------------------------------


def _disjoint_regions(regions):
    """Resolve partially-overlapping (outer, holes) regions into disjoint ones.

    Uses shapely's 2D union when available; without it, regions pass through
    unchanged (correct for non-overlapping art, e.g. plain text).
    """
    if len(regions) <= 1:
        return regions
    try:
        from shapely import make_valid
        from shapely.geometry import MultiPolygon, Polygon
        from shapely.ops import unary_union
    except ImportError:
        return regions
    # SVG subpaths may self-intersect; repair each region before the union.
    merged = unary_union([make_valid(Polygon(o, holes)) for o, holes in regions])
    merged = merged.buffer(0)
    if merged.is_empty:
        return []
    polys = merged.geoms if isinstance(merged, MultiPolygon) else [merged]
    out = []
    for p in polys:
        outer = list(p.exterior.coords)[:-1]
        holes = [list(r.coords)[:-1] for r in p.interiors]
        out.append((outer, holes))
    return out


def _concat_solids(solids):
    """Combine disjoint solids into one multi-shell solid (no boolean)."""
    if len(solids) == 1:
        return solids[0]
    verts: list[float] = []
    tris: list[int] = []
    for s in solids:
        m = s.mesh()
        base = len(verts) // 3
        verts.extend(m.vertices_flat())
        tris.extend(i + base for i in m.triangles_flat())
    return ist.solid_from_mesh(verts, tris)


def _circle_of_loop(loop, rel_tol=1e-6):
    """If a 2D loop is a discretized circle, return (center, radius)."""
    if len(loop) < 8:
        return None
    cx = sum(p[0] for p in loop) / len(loop)
    cy = sum(p[1] for p in loop) / len(loop)
    dists = [math.hypot(p[0] - cx, p[1] - cy) for p in loop]
    r = sum(dists) / len(dists)
    if r <= 0:
        return None
    if max(abs(d - r) for d in dists) > rel_tol * r:
        return None
    return (cx, cy), r


def _extrude_region(placement, outer, holes, start, end):
    ow = _loop_to_wire(placement, outer, z=start)
    if holes:
        face = ist.make_face_with_holes(ow, [_loop_to_wire(placement, h, z=start) for h in holes])
    else:
        face = ist.make_face(ow)
    vec = ist.Pnt(*_vscale(placement.normal(), end - start))
    solid = ist.make_prism(face, vec)
    # Circular loops sweep exact cylindrical walls: stamp provenance so STEP
    # export can emit a real CYLINDRICAL_SURFACE (bosses and drilled holes).
    stamp = getattr(solid, "add_cylinder_hint", None)  # ironstream >= 0.2
    if stamp is not None:
        normal = placement.normal()
        for loop in [outer, *holes]:
            circ = _circle_of_loop(loop)
            if circ:
                (cx, cy), r = circ
                world = placement.to_world(cx, cy, start)
                stamp(ist.Pnt(*world), ist.Pnt(*normal), r)
    return solid


def _extrusion(ctx, node):
    shapes = _shapes(ctx, node, "shape")
    placement: Placement = ctx.get(node, "sketch")
    start = float(ctx.get(node, "start"))
    end = float(ctx.get(node, "end"))
    cut = bool(ctx.get(node, "cut"))

    if len(shapes) == 1 and isinstance(shapes[0], Loops):
        # multi-loop region (e.g. SVG art or text): islands + holes. Regions
        # are made disjoint in 2D first (partially-overlapping subpaths would
        # otherwise feed overlapping coplanar prisms to the boolean engine),
        # so the extruded solids can simply be concatenated.
        regions = _disjoint_regions(classify_loops(shapes[0].loops))
        solids = [
            _extrude_region(placement, outer, holes, start, end)
            for outer, holes in regions
        ]
        solids = [s for s in solids if abs(s.volume()) > 1e-6]
        if not solids:
            raise UnsupportedNodeError("extrusion region had no usable loops")
        return OpResult(_concat_solids(solids), cut)

    solid = _extrude_region(placement, shapes[0], shapes[1:], start, end)
    return OpResult(solid, cut)


def _bbox(solid):
    """Axis-aligned bounds ((minx,miny,minz),(maxx,maxy,maxz)) from a solid mesh."""
    v = solid.mesh().vertices_flat()
    xs, ys, zs = v[0::3], v[1::3], v[2::3]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _clip_profile_to_axis(pts3d, loc, perp):
    """Clip a revolution profile to one side of its axis.

    A profile that straddles the axis (e.g. an ellipse centred on it) would be
    double-covered by a 360° revolution and cancel to ~zero volume. We keep the
    dominant half-plane (normal ``perp``, through ``loc``) via Sutherland-
    Hodgman; edges crossing the axis get an intersection point (radius 0 on the
    axis). A profile already on one side is returned essentially unchanged.
    """
    def sd(p):
        return sum((p[k] - loc[k]) * perp[k] for k in range(3))

    ds = [sd(p) for p in pts3d]
    pos = sum(d for d in ds if d > 0)
    neg = sum(-d for d in ds if d < 0)
    if min(pos, neg) < 1e-9:
        return pts3d  # already one-sided
    keep_pos = pos >= neg
    inside = (lambda d: d >= -1e-9) if keep_pos else (lambda d: d <= 1e-9)
    n = len(pts3d)
    out = []
    for i in range(n):
        a, da = pts3d[i], ds[i]
        b, db = pts3d[(i + 1) % n], ds[(i + 1) % n]
        if inside(da):
            out.append(a)
        if (da > 0) != (db > 0) and abs(da - db) > 1e-12:
            t = da / (da - db)
            out.append(tuple(a[k] + (b[k] - a[k]) * t for k in range(3)))
    return out


def _lathe(ctx, node, segments=64):
    placement: Placement = ctx.get(node, "sketch")
    loop = _shapes(ctx, node, "shape")[0]
    _, loc, direction = ctx.get(node, "axis")
    cut = bool(ctx.get(node, "cut"))
    pts3d = [placement.to_world(x, y, 0.0) for (x, y) in loop]
    # in-plane direction perpendicular to the axis (the "radius" direction)
    perp = _norm(_cross(_norm(direction), placement.normal()))
    pts3d = _clip_profile_to_axis(pts3d, loc, perp)
    wire = ist.make_polygon([ist.Pnt(*p) for p in pts3d])
    axis = ist.Ax1(ist.Pnt(*loc), ist.Pnt(*direction))
    solid = ist.make_revol(wire, axis, 2 * math.pi, ist.MeshParams(segments, segments))
    return OpResult(solid, cut)


def _sphere(ctx, node, segments=48):
    cx, cy = ctx.get(node, "center")
    r = float(ctx.get(node, "radius"))
    placement: Placement = ctx.get(node, "sketch")
    cut = bool(ctx.get(node, "cut"))
    solid = ist.make_sphere(r, ist.MeshParams(segments, segments))
    world_center = placement.to_world(cx, cy, 0.0)
    solid = ist.transform(solid, ist.Trsf.translation(ist.Pnt(*world_center)))
    return OpResult(solid, cut)


# --- transform operations (wrap another operation's solid) -----------------


def _child_op(ctx, node):
    for key in ("shape", "operation", "solid", "body", "surface"):
        v = ctx.get_opt(node, key)
        if isinstance(v, OpResult):
            return v
    return None


def _mirror(ctx, node):
    child = _child_op(ctx, node)
    if child is None:
        return METADATA
    # reflect across the YZ plane (x -> -x), matching the Rust reference
    return OpResult(ist.transform(child.solid, ist.Trsf.scale_xyz(-1.0, 1.0, 1.0)), child.cut)


def _scale(ctx, node):
    child = _child_op(ctx, node)
    if child is None:
        return METADATA
    f = float(ctx.get(node, "factor"))
    return OpResult(ist.transform(child.solid, ist.Trsf.scale_uniform(f)), child.cut)


def _shell(ctx, node):
    child = _child_op(ctx, node)
    t = float(ctx.get(node, "thickness"))
    (mnx, mny, mnz), (mxx, mxy, mxz) = _bbox(child.solid)
    dx, dy, dz = (mxx - mnx) - 2 * t, (mxy - mny) - 2 * t, (mxz - mnz) - t
    if min(dx, dy, dz) <= 0:
        return child  # too thin to hollow
    inner = ist.make_box(ist.Pnt(mnx + t, mny + t, mnz + t), dx, dy, dz)  # open top
    return OpResult(ist.cut(child.solid, inner), child.cut)


def _stitch(ctx, node):
    solids = [o.solid for o in ctx.get_list(node, "surfaces") if isinstance(o, OpResult)]
    if not solids:
        return METADATA
    return OpResult(ist.fuse_all(solids), False)


# --- sheet metal -----------------------------------------------------------


@dataclass
class SmBody(OpResult):
    thickness: float = 2.0
    placement: "Placement | None" = None


def _sm_base_flange(ctx, node):
    loop = _shapes(ctx, node, "profile")[0]
    placement: Placement = ctx.get(node, "sketch")
    t = float(ctx.get(node, "thickness"))
    direction = ctx.get_opt(node, "direction") or node.params.get("direction") or "positive"
    z0 = {"negative": -t, "both": -t / 2.0}.get(direction, 0.0)
    face = ist.make_face(_loop_to_wire(placement, loop, z0))
    solid = ist.make_prism(face, ist.Pnt(*_vscale(placement.normal(), t)))
    return SmBody(solid, False, t, placement)


def _sm_edge_flange(ctx, node):
    body = ctx.get(node, "body")
    t = getattr(body, "thickness", 2.0)
    length = float(ctx.get_opt(node, "length", 10.0) or 10.0)
    angle = math.radians(float(ctx.get_opt(node, "bend_angle", 90.0) or 90.0))
    (mnx, mny, mnz), (mxx, mxy, mxz) = _bbox(body.solid)
    dx, dy = mxx - mnx, mxy - mny
    fz = length * math.sin(angle) or length
    if dy >= dx:
        flange = ist.make_box(ist.Pnt(mxx - t, mny, mxz), t, dy, fz)
    else:
        flange = ist.make_box(ist.Pnt(mnx, mxy - t, mxz), dx, t, fz)
    return SmBody(ist.fuse(body.solid, flange), False, t, getattr(body, "placement", None))


def _sm_to_solid(ctx, node):
    body = ctx.get(node, "body")
    return OpResult(body.solid, getattr(body, "cut", False))


def _sm_passthrough(ctx, node):
    """Bends/tabs/hems/unfold not modelled by the reference kernel: forward the
    upstream body unchanged (approximate geometry, matching ironstream-dag)."""
    body = ctx.get_opt(node, "body") or ctx.get_opt(node, "surface")
    return body if body is not None else METADATA


# --- not modelled by the reference kernel: pass the solid through ----------


def _passthrough_solid(ctx, node):
    child = _child_op(ctx, node)
    return child if child is not None else METADATA


# --- 3D points / curves ----------------------------------------------------


def _point3d(ctx, node):
    return (float(ctx.get(node, "x")), float(ctx.get(node, "y")), float(ctx.get(node, "z")))


def _spline(ctx, node):
    # Linear approximation through the control points (2D sketch loop or 3D).
    return list(ctx.get_list(node, "points"))


def _area_signed(loop):
    a = 0.0
    for i in range(len(loop)):
        x1, y1 = loop[i]
        x2, y2 = loop[(i + 1) % len(loop)]
        a += x1 * y2 - x2 * y1
    return a / 2.0


def _polyline_area(loop):
    return abs(_area_signed(loop))


def _contains(loop, pt):
    """Even-odd point-in-polygon (ray cast toward +x)."""
    x, y = pt
    inside = False
    n = len(loop)
    for i in range(n):
        x1, y1 = loop[i]
        x2, y2 = loop[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if xi > x:
                inside = not inside
    return inside


@dataclass
class Loops:
    """Several closed 2D loops making one region (islands + holes)."""

    loops: list


def classify_loops(loops):
    """Group loops into (outer, [holes]) pairs by even-odd containment depth."""
    loops = [l for l in loops if len(l) >= 3 and _polyline_area(l) > 1e-9]
    order = sorted(range(len(loops)), key=lambda i: -_polyline_area(loops[i]))
    depth = {}
    for i in order:
        depth[i] = sum(
            1 for j in order
            if j != i
            and _polyline_area(loops[j]) > _polyline_area(loops[i])
            and _contains(loops[j], loops[i][0])
        )
    regions = []
    for o in order:
        if depth[o] % 2 != 0:
            continue
        holes = [
            loops[h] for h in order
            if depth[h] == depth[o] + 1 and _contains(loops[o], loops[h][0])
        ]
        regions.append((loops[o], holes))
    return regions


def _dedupe(loop, tol=1e-9):
    out = []
    for p in loop:
        if not out or (p[0] - out[-1][0]) ** 2 + (p[1] - out[-1][1]) ** 2 > tol:
            out.append(p)
    if len(out) > 1 and (out[0][0] - out[-1][0]) ** 2 + (out[0][1] - out[-1][1]) ** 2 <= tol:
        out.pop()
    return out


def _svg_path_loops(svg):
    """Sample every closed subpath of every <path d=...> in the SVG."""
    import re

    from svg.path import Move, parse_path

    loops = []
    for d in re.findall(r"""d=["']([^"']+)["']""", svg):
        # split the path into subpaths at Move commands
        subpaths, current = [], []
        for seg in parse_path(d):
            if isinstance(seg, Move):
                if current:
                    subpaths.append(current)
                current = []
            else:
                current.append(seg)
        if current:
            subpaths.append(current)
        for segs in subpaths:
            total = sum(s.length() for s in segs)
            if total <= 0:
                continue
            n = max(24, min(240, int(total)))
            pts = []
            for s in segs:
                k = max(2, int(n * s.length() / total))
                pts.extend(
                    (s.point(t / k).real, s.point(t / k).imag) for t in range(k)
                )
            loop = _dedupe(pts)
            if len(loop) >= 3:
                loops.append(loop)
    return loops


def _svg_text_loops(svg):
    """Render every <text> element to glyph-outline loops (matplotlib fonts)."""
    import re

    texts = re.findall(r"<text([^>]*)>([^<]*)</text>", svg)
    if not texts:
        return []

    try:
        from matplotlib.font_manager import FontProperties
        from matplotlib.textpath import TextPath
    except ImportError as e:  # pragma: no cover
        raise UnsupportedNodeError(
            "SVGShape contains <text>: glyph outlines need matplotlib "
            "(install castiron[svg])"
        ) from e

    loops = []
    for attrs, content in texts:
        content = content.strip()
        if not content:
            continue
        def attr(name, default):
            m = re.search(rf"""{name}=["']([^"']+)["']""", attrs)
            return m.group(1) if m else default
        x0 = float(attr("x", "0"))
        y0 = float(attr("y", "0"))
        size = float(attr("font-size", "16"))
        family = attr("font-family", "sans-serif")
        try:
            prop = FontProperties(family=family)
        except Exception:
            prop = FontProperties()
        tp = TextPath((0, 0), content, size=size, prop=prop)
        for poly in tp.to_polygons():
            # TextPath is y-up; SVG is y-down with (x0, y0) the baseline.
            loop = _dedupe([(x0 + px, y0 - py) for px, py in poly])
            if len(loop) >= 3:
                loops.append(loop)
    return loops


def _svg_shape(ctx, node):
    """Flatten an inline SVG (paths + text) into placed 2D loops.

    Beziers/arcs are sampled via ``svg.path``; ``<text>`` is rendered to glyph
    outlines. SVG's y-axis points down, so loops are flipped, then scale,
    rotation and shift are applied in sketch-local space. Returns a
    :class:`Loops` (islands and holes are classified downstream).
    """
    svg = ctx.get(node, "svg")
    scale = float(ctx.get_opt(node, "scale", 1.0) or 1.0)
    angle = float(ctx.get_opt(node, "angle", 0.0) or 0.0)
    xshift = float(ctx.get_opt(node, "xshift", 0.0) or 0.0)
    yshift = float(ctx.get_opt(node, "yshift", 0.0) or 0.0)

    loops = _svg_path_loops(svg) + _svg_text_loops(svg)
    if not loops:
        raise UnsupportedNodeError("SVGShape had no usable path or text data")

    # Match the reference kernel semantics (kernel-replicad SVGShape):
    # scale (+ SVG y-flip), center on the bounding box, rotate, then shift.
    scaled = [[(x * scale, -y * scale) for x, y in loop] for loop in loops]
    xs = [x for loop in scaled for x, _ in loop]
    ys = [y for loop in scaled for _, y in loop]
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0

    ca, sa = math.cos(angle), math.sin(angle)
    placed = []
    for loop in scaled:
        out = []
        for x, y in loop:
            sx, sy = x - cx, y - cy
            out.append((sx * ca - sy * sa + xshift, sx * sa + sy * ca + yshift))
        placed.append(out)
    return Loops(placed)


# --- loft / sweep (mesh skinning via ironstream.solid_from_mesh) -----------


def _loft(ctx, node):
    shapes_key = "shapes" if "shapes" in node.deps else "profiles"
    sketch_key = "sketchs" if "sketchs" in node.deps else "sketches"
    profiles = ctx.get_list(node, shapes_key)
    placements = ctx.get_list(node, sketch_key)
    sections = []
    for i, loop in enumerate(profiles):
        pl = placements[i] if i < len(placements) else Placement()
        sections.append([pl.to_world(x, y, 0.0) for (x, y) in loop])
    cut = bool(ctx.get_opt(node, "cut", False) or False)
    return OpResult(loft_solid(sections), cut)


def _sweep(ctx, node):
    profile = _shapes(ctx, node, "profile")[0]  # 2D sketch-local loop
    path = ctx.get(node, "path")  # Spline3D / Helix3D -> list of 3D control points
    cut = bool(ctx.get_opt(node, "cut", False) or False)
    return OpResult(sweep_solid(profile, path), cut)


def _helix3d(ctx, node, seg_per_turn=32, max_points=400):
    cx, cy, cz = ctx.get(node, "center")
    axis = _norm(ctx.get(node, "dir"))
    height = float(ctx.get(node, "height"))
    pitch = float(ctx.get(node, "pitch"))
    radius = float(ctx.get(node, "radius"))
    sign = -1.0 if bool(ctx.get_opt(node, "lefthand", False)) else 1.0
    ref = (0.0, 0.0, 1.0) if abs(axis[2]) < 0.95 else (1.0, 0.0, 0.0)
    e1 = _norm(_cross(ref, axis))
    e2 = _norm(_cross(axis, e1))
    turns = height / pitch if pitch else 1.0
    # Cap the total point count: a many-turn helix drives a Sweep that is then
    # boolean'd, and the BSP cost is superlinear in triangle count. This keeps a
    # 10-turn thread from exploding to tens of thousands of triangles.
    npts = max(2, min(max_points, int(turns * seg_per_turn)))
    pts = []
    for i in range(npts + 1):
        t = turns * i / npts
        a = sign * 2 * math.pi * t
        z = height * i / npts
        pts.append(tuple(
            (cx, cy, cz)[k] + radius * (math.cos(a) * e1[k] + math.sin(a) * e2[k]) + z * axis[k]
            for k in range(3)
        ))
    return pts


# --- composite nodes (foundation @expand pattern) ---------------------------
# Many foundation convenience types carry a `result` dep pointing at the
# primitive node they expand to (Box -> Extrusion, Torus -> Lathe,
# RegularPolygon -> Polygon, ...). Their handler is pure delegation.


def _result(ctx, node):
    return ctx.get(node, "result")


# All classes in cadbuildr.foundation.gen.models whose value is their `result`.
_RESULT_NODES = [
    "ArcFromTwoPointsAndRadius", "Box", "CenterArc", "Cone", "ConvexHull",
    "CounterBoreHole", "CounterSinkHole", "Cylinder", "EllipticalCenterArc",
    "FilletPolyline", "Helix2D", "JernArc", "Polyline", "RadiusArc",
    "RectangleFrom2Points", "RectangleFromCenterAndSides", "RectangleRounded",
    "RegularPolygon", "SagittaArc", "SlotCenterPoint", "SlotCenterToCenter",
    "SlotOverall", "SquareFromCenterAndSide", "TangentArc", "TappedHole",
    "Thread", "ThreePointArc", "Torus", "Trapezoid", "Triangle",
]


# --- remaining curve / shape / op nodes -------------------------------------


def _bezier(ctx, node, samples=32):
    """De Casteljau sampling of a Bezier through its control points."""
    pts = ctx.get_list(node, "points")
    out = []
    for i in range(samples + 1):
        t = i / samples
        work = list(pts)
        while len(work) > 1:
            work = [
                tuple(a[k] + (b[k] - a[k]) * t for k in range(len(a)))
                for a, b in zip(work, work[1:])
            ]
        out.append(work[0])
    return out


def _bspline(ctx, node, samples=64):
    """Clamped uniform B-spline sampled with de Boor's algorithm."""
    pts = ctx.get_list(node, "points")
    degree = int(ctx.get_opt(node, "degree", 3) or 3)
    n = len(pts)
    p = min(degree, n - 1)
    if p < 1:
        return list(pts)
    # clamped uniform knot vector
    m = n + p + 1
    knots = [0.0] * (p + 1) + [i / (n - p) for i in range(1, n - p)] + [1.0] * (p + 1)
    assert len(knots) == m

    def de_boor(t):
        # find span k with knots[k] <= t < knots[k+1]
        k = max(p, min(n - 1, next(i for i in range(len(knots) - 1) if t < knots[i + 1] or i == n - 1)))
        d = [pts[j] for j in range(k - p, k + 1)]
        for r in range(1, p + 1):
            for j in range(p, r - 1, -1):
                i = k - p + j
                den = knots[i + p - r + 1] - knots[i]
                alpha = 0.0 if den == 0 else (t - knots[i]) / den
                d[j] = tuple(
                    (1 - alpha) * d[j - 1][c] + alpha * d[j][c] for c in range(len(d[j]))
                )
        return d[p]

    return [de_boor(min(i / samples, 1.0 - 1e-12)) for i in range(samples + 1)]


def _ellipse_arc(ctx, node, samples=32):
    cx, cy = ctx.get(node, "center")
    a = float(ctx.get(node, "a"))
    b = float(ctx.get(node, "b"))
    t0 = float(ctx.get(node, "start_angle"))
    t1 = float(ctx.get(node, "end_angle"))
    return [(cx + a * math.cos(t0 + (t1 - t0) * i / samples),
             cy + b * math.sin(t0 + (t1 - t0) * i / samples)) for i in range(samples + 1)]


def _offset2d(ctx, node, samples_per_arc=8):
    """2D offset of a closed shape (shapely buffer)."""
    loop = _shapes(ctx, node, "shape")[0]
    dist = float(ctx.get(node, "distance"))
    try:
        from shapely.geometry import Polygon as SPoly
    except ImportError as e:
        raise UnsupportedNodeError("Offset2D needs shapely (castiron[svg])") from e
    buffered = SPoly(loop).buffer(dist)
    if buffered.is_empty:
        raise UnsupportedNodeError("Offset2D collapsed the shape to nothing")
    return list(buffered.exterior.coords)[:-1]


def _trace(ctx, node):
    """A constant-width path: buffer the polyline by width/2."""
    pts = ctx.get_list(node, "path_points")
    width = float(ctx.get(node, "width"))
    try:
        from shapely.geometry import LineString
    except ImportError as e:
        raise UnsupportedNodeError("Trace needs shapely (castiron[svg])") from e
    buf = LineString([p[:2] for p in pts]).buffer(width / 2.0)
    return list(buf.exterior.coords)[:-1]


def _text(ctx, node):
    """3D text: render glyph outlines (matplotlib) as a multi-loop region."""
    text = str(ctx.get(node, "text"))
    size = float(ctx.get_opt(node, "size", 16.0) or 16.0)
    xshift = float(ctx.get_opt(node, "xshift", 0.0) or 0.0)
    yshift = float(ctx.get_opt(node, "yshift", 0.0) or 0.0)
    try:
        from matplotlib.textpath import TextPath
    except ImportError as e:
        raise UnsupportedNodeError("Text needs matplotlib (castiron[svg])") from e
    tp = TextPath((0, 0), text, size=size)
    loops = []
    for poly in tp.to_polygons():
        loop = _dedupe([(xshift + px, yshift + py) for px, py in poly])
        if len(loop) >= 3:
            loops.append(loop)
    if not loops:
        raise UnsupportedNodeError("Text produced no glyph outlines")
    return Loops(loops)


def _wedge(ctx, node):
    """OCCT-style wedge: a box whose top face is shortened to ltx in x."""
    from .loft import loft_solid

    cx, cy = ctx.get(node, "center")
    dx = float(ctx.get(node, "dx"))
    dy = float(ctx.get(node, "dy"))
    dz = float(ctx.get(node, "dz"))
    ltx = float(ctx.get_opt(node, "ltx", 0.0) or 0.0)
    x0, y0 = cx - dx / 2.0, cy - dy / 2.0
    bottom = [(x0, y0, 0.0), (x0 + dx, y0, 0.0), (x0 + dx, y0 + dy, 0.0), (x0, y0 + dy, 0.0)]
    tx = max(ltx, dx * 1e-4)  # degenerate top edge -> thin sliver, keeps mesh valid
    top = [(x0, y0, dz), (x0 + tx, y0, dz), (x0 + tx, y0 + dy, dz), (x0, y0 + dy, dz)]
    return OpResult(loft_solid([bottom, top]), False)


def _sm_body(ctx, node):
    return ctx.get(node, "base")


# Enum/option types from the schema; they appear as params, never as geometry.
_ENUM_TYPES = {
    "BendPosition", "CornerReliefType", "CornerType", "DimensionPosition",
    "FlangePosition", "HemType", "MaterialOptions", "ReliefType", "SheetDirection",
}

# Construction / selection nodes that carry no geometry.
_IGNORED = {
    "EdgeFinder", "IsCircleRule", "OfTypeRule", "InPlaneFinderRule", "AtAngleFinderRule",
    "AtDistanceFinderRule", "ContainsPointFinderRule", "InBoxFinderRule", "InDirectionFinderRule",
    "AndFinderRule", "EitherFinderRule", "LengthRangeRule", "RadiusRangeRule", "ParallelToAxisRule",
    "OnFaceRule", "SortByRule", "Color", "BoundingBox", "FixedTranslationConstraint",
    "AssemblyInterface", "InterfaceGridSpec", "EdgeRef", "EdgeReference",
}


def _ignore(ctx, node):
    return METADATA


def _part(ctx, node):
    ops = [op for op in ctx.get_list(node, "operations") if isinstance(op, OpResult)]
    solid = None
    i = 0
    n = len(ops)
    while i < n:
        cut = ops[i].cut
        # Batch a run of consecutive same-direction ops into ONE boolean:
        # A ∪ (B ∪ C) and A \ (B ∪ C) both equal the sequential form, but the
        # union of the tools is a cheap concat when they're disjoint (the common
        # drilled-holes / bolt-pattern case), so this collapses N BSP passes to
        # one and avoids the mesh-fragmentation blowup.
        j = i
        tools = []
        while j < n and ops[j].cut == cut:
            tools.append(ops[j].solid)
            j += 1
        combined = tools[0] if len(tools) == 1 else ist.fuse_all(tools)
        if solid is None:
            if not cut:
                solid = combined  # leading cuts with no base are ignored
        elif cut:
            solid = ist.cut(solid, combined)
        else:
            solid = ist.fuse(solid, combined)
        i = j
    if solid is None:
        return None  # empty part (e.g. only a BoundingBox); skipped by assemblies
    name = ctx.get_opt(node, "name") or "part"
    return PartResult(str(name), solid)


def _assembly(ctx, node):
    """Collect descendant parts. Geometry is already world-placed because each
    part's sketch frames chain (via ``top_frame``) up to the positioned
    assembly frames, so no extra instance transform is applied here."""
    parts: list[PartResult] = []
    for comp in ctx.get_list(node, "components"):
        if isinstance(comp, PartResult):
            parts.append(comp)
        elif isinstance(comp, list):
            parts.extend(p for p in comp if isinstance(p, PartResult))
    return parts


HANDLERS = {
    "FloatParameter": _param,
    "IntParameter": _param,
    "StringParameter": _param,
    "BoolParameter": _param,
    "Material": _material,
    "Point": _point,
    "SketchOrigin": _point,
    "Line": _line,
    "Arc": _arc,
    "Polygon": _polygon,
    "Rectangle": _polygon,
    "Square": _polygon,
    "Hexagon": _polygon,
    "CustomClosedShape": _polygon,
    "CustomOpenShape": _polygon,
    "Circle": _circle,
    "Ellipse": _ellipse,
    "SVGShape": _svg_shape,
    "Frame": _frame,
    "Plane": _plane,
    "Sketch": _sketch,
    "Axis": _axis,
    "Point3D": _point3d,
    "Spline": _spline,
    "Spline3D": _spline,
    "Extrusion": _extrusion,
    "Hole": lambda ctx, node: OpResult(_extrusion(ctx, node).solid, True),
    "Lathe": _lathe,
    "Sphere": _sphere,
    "Mirror": _mirror,
    "Scale": _scale,
    "Shell": _shell,
    "Stitch": _stitch,
    "Loft": _loft,
    "SurfaceLoft": _loft,
    "MultiSectionSweep": _loft,
    "Sweep": _sweep,
    "Helix3D": _helix3d,
    # not modelled by the reference kernel -> forward the solid unchanged
    "Fillet": _passthrough_solid,
    "Chamfer": _passthrough_solid,
    "Draft": _passthrough_solid,
    "FullRound": _passthrough_solid,
    "Thicken": _sm_passthrough,
    # sheet metal
    "SheetMetalBaseFlange": _sm_base_flange,
    "SheetMetalEdgeFlange": _sm_edge_flange,
    "SheetMetalToSolid": _sm_to_solid,
    "SheetMetalTab": _sm_passthrough,
    "SheetMetalHem": _sm_passthrough,
    "SheetMetalMiterFlange": _sm_passthrough,
    "SheetMetalBend": _sm_passthrough,
    "SheetMetalContourFlange": _sm_passthrough,
    "SheetMetalCornerSeam": _sm_passthrough,
    "SheetMetalSketchedBend": _sm_passthrough,
    "SheetMetalClosedCorner": _sm_passthrough,
    "Unfold": _sm_passthrough,
    "Part": _part,
    "PartRoot": _part,
    "Assembly": _assembly,
    "AssemblyRoot": _assembly,
    # curves / shapes
    "Bezier": _bezier,
    "BSpline": _bspline,
    "EllipseArc": _ellipse_arc,
    "Offset2D": _offset2d,
    "Trace": _trace,
    "Text": _text,
    "Wedge": _wedge,
    # ops the reference kernel forwards unchanged
    "Project": _passthrough_solid,
    "Section": _passthrough_solid,
    "Split": _passthrough_solid,
    # sheet metal extras (forwarded like the other bend variants)
    "SheetMetalBody": _sm_body,
    "SheetMetalCornerRelief": _sm_passthrough,
    "SheetMetalFold": _sm_passthrough,
    "SheetMetalJog": _sm_passthrough,
    "SheetMetalLoftedBend": _sm_passthrough,
    # composite nodes: value == their expanded `result`
    **{t: _result for t in _RESULT_NODES},
    **{t: _ignore for t in _ENUM_TYPES},
    **{t: _ignore for t in _IGNORED},
}
