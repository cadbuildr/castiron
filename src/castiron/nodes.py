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


def _extrusion(ctx, node):
    shapes = _shapes(ctx, node, "shape")
    placement: Placement = ctx.get(node, "sketch")
    start = float(ctx.get(node, "start"))
    end = float(ctx.get(node, "end"))
    cut = bool(ctx.get(node, "cut"))

    outer = _loop_to_wire(placement, shapes[0], z=start)
    if len(shapes) > 1:
        holes = [_loop_to_wire(placement, loop, z=start) for loop in shapes[1:]]
        face = ist.make_face_with_holes(outer, holes)
    else:
        face = ist.make_face(outer)
    vec = ist.Pnt(*_vscale(placement.normal(), end - start))
    return OpResult(ist.make_prism(face, vec), cut)


def _bbox(solid):
    """Axis-aligned bounds ((minx,miny,minz),(maxx,maxy,maxz)) from a solid mesh."""
    v = solid.mesh().vertices_flat()
    xs, ys, zs = v[0::3], v[1::3], v[2::3]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _lathe(ctx, node, segments=64):
    placement: Placement = ctx.get(node, "sketch")
    loop = _shapes(ctx, node, "shape")[0]
    _, loc, direction = ctx.get(node, "axis")
    cut = bool(ctx.get(node, "cut"))
    wire = _loop_to_wire(placement, loop, 0.0)
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


def _polyline_area(loop):
    a = 0.0
    for i in range(len(loop)):
        x1, y1 = loop[i]
        x2, y2 = loop[(i + 1) % len(loop)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def _svg_shape(ctx, node, samples=240):
    """Flatten the largest subpath of an inline SVG into a placed 2D loop.

    Beziers/arcs are sampled via ``svg.path``. SVG's y-axis points down, so we
    flip it, then apply scale, rotation, and shift into sketch-local space.
    """
    import re

    from svg.path import parse_path

    svg = ctx.get(node, "svg")
    scale = float(ctx.get_opt(node, "scale", 1.0) or 1.0)
    angle = float(ctx.get_opt(node, "angle", 0.0) or 0.0)
    xshift = float(ctx.get_opt(node, "xshift", 0.0) or 0.0)
    yshift = float(ctx.get_opt(node, "yshift", 0.0) or 0.0)

    loops = []
    for d in re.findall(r"""d=["']([^"']+)["']""", svg):
        path = parse_path(d)
        if path.length() == 0:
            continue
        loop = [(p.real, p.imag) for p in (path.point(i / samples) for i in range(samples))]
        loops.append(loop)
    if not loops:
        if "<text" in svg:
            raise UnsupportedNodeError(
                "SVGShape contains <text>; rendering text to glyph outlines needs a "
                "font engine (not yet supported) — use <path> data instead"
            )
        raise UnsupportedNodeError("SVGShape had no usable path data")
    loop = max(loops, key=_polyline_area)

    ca, sa = math.cos(angle), math.sin(angle)
    out = []
    for x, y in loop:
        sx, sy = x * scale, -y * scale  # flip SVG y
        out.append((sx * ca - sy * sa + xshift, sx * sa + sy * ca + yshift))
    return out


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


def _helix3d(ctx, node, seg_per_turn=32):
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
    npts = max(2, int(turns * seg_per_turn))
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
    for op in ops:
        if solid is None:
            if not op.cut:
                solid = op.solid
        elif op.cut:
            solid = ist.cut(solid, op.solid)
        else:
            solid = ist.fuse(solid, op.solid)
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
    **{t: _ignore for t in _IGNORED},
}
