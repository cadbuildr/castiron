"""Edge blends (Fillet / Chamfer) on IronStream mesh solids.

IronStream solids are boundary meshes without B-rep edge topology, so blends
are built geometrically:

1. **Feature edges.** Each triangle edge registers an interval on its
   supporting line, tagged with the triangle's normal and the side it lies on.
   Where two triangles with clearly different normals cover the same stretch of
   a line, that stretch is a sharp edge. Working per line (not per vertex pair)
   makes this robust to the T-junctions boolean output leaves.
2. **Edges.** Collinear stretches with the same face pair merge into straight
   edges; chains of short edges that close into a planar circle become circular
   edges (a cylinder meeting a cap).
3. **Selection.** foundation's ``EdgeFinder`` rules filter the edges with the
   semantics of the OCCT kernel's ``applyEdgeFinder`` / ``postFilterEdges``.
4. **Tools.** For each convex selected edge, the material between the edge and
   the blend (the fillet's concave arc, or the chamfer's flat) is built as a 2D
   section in the plane perpendicular to the edge and swept: extruded along a
   straight edge, revolved about a circular edge's axis. All tools are cut from
   the solid in one boolean.

Corners where several filleted edges meet come out as the intersection of the
edge roll-offs rather than OCCT's spherical/blended vertex patch, and arcs that
are not full circles, concave edges and freeform edges are not blended; they
are reported in ``BlendResult.skipped``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import ironstream as ist

FEATURE_ANGLE = math.radians(20.0)
ANGULAR_STEP = 2 * math.pi / 64


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _len(a):
    return math.sqrt(_dot(a, a))


def _unit(a):
    n = _len(a)
    return (a[0] / n, a[1] / n, a[2] / n) if n > 0 else a


@dataclass
class Edge:
    kind: str  # "LINE" | "CIRCLE"
    a: tuple  # LINE: endpoints; CIRCLE: a point on the circle
    b: tuple
    n1: tuple  # outward normals of the two faces (LINE: exact; CIRCLE: at `a`)
    n2: tuple
    t1: tuple = (0.0, 0.0, 0.0)  # direction into face 1, perpendicular to the edge
    t2: tuple = (0.0, 0.0, 0.0)
    center: tuple = (0.0, 0.0, 0.0)
    axis: tuple = (0.0, 0.0, 1.0)
    radius: float = 0.0
    points: list = field(default_factory=list)

    @property
    def length(self):
        return 2 * math.pi * self.radius if self.kind == "CIRCLE" else _len(_sub(self.b, self.a))

    @property
    def midpoint(self):
        return self.a if self.kind == "CIRCLE" else _mul(_add(self.a, self.b), 0.5)

    @property
    def direction(self):
        return self.axis if self.kind == "CIRCLE" else _unit(_sub(self.b, self.a))

    @property
    def convex(self):
        return _dot(self.t1, self.n2) < -1e-9 and _dot(self.t2, self.n1) < -1e-9


# --------------------------------------------------------------- extraction


def _segments(mesh, scale):
    """Sharp stretches of the mesh: [(p, q, n1, t1, n2, t2)]."""
    v = mesh.vertices_flat()
    t = mesh.triangles_flat()
    P = [(v[i], v[i + 1], v[i + 2]) for i in range(0, len(v), 3)]
    q = 1e-6 * scale
    lines: dict = {}
    for k in range(0, len(t), 3):
        ia, ib, ic = t[k], t[k + 1], t[k + 2]
        a, b, c = P[ia], P[ib], P[ic]
        n = _cross(_sub(b, a), _sub(c, a))
        if _len(n) < 1e-14 * scale * scale:
            continue
        n = _unit(n)
        for p0, p1, p2 in ((a, b, c), (b, c, a), (c, a, b)):
            d = _sub(p1, p0)
            L = _len(d)
            if L < q:
                continue
            d = _unit(d)
            # canonical direction sign (first non-zero component positive)
            sgn = 1.0
            for comp in d:
                if abs(comp) > 1e-9:
                    sgn = 1.0 if comp > 0 else -1.0
                    break
            dc = _mul(d, sgn)
            foot = _sub(p0, _mul(dc, _dot(p0, dc)))
            key = (
                round(dc[0], 6), round(dc[1], 6), round(dc[2], 6),
                round(foot[0] / q) if q else 0, round(foot[1] / q) if q else 0, round(foot[2] / q) if q else 0,
            )
            s0, s1 = _dot(p0, dc), _dot(p1, dc)
            # direction from the edge into this triangle, perpendicular to the edge
            off = _sub(p2, p0)
            side = _unit(_sub(off, _mul(dc, _dot(off, dc))))
            ent = lines.setdefault(key, (dc, foot, []))
            ent[2].append((min(s0, s1), max(s0, s1), n, side))
    out = []
    for dc, foot, ivs in lines.values():
        if len(ivs) < 2:
            continue
        cuts = sorted({x for iv in ivs for x in iv[:2]})
        for lo, hi in zip(cuts, cuts[1:]):
            if hi - lo < q:
                continue
            mid = 0.5 * (lo + hi)
            cover = [iv for iv in ivs if iv[0] <= mid <= iv[1]]
            if len(cover) != 2:
                continue
            (_, _, na, sa), (_, _, nb, sb) = cover
            if math.acos(max(-1.0, min(1.0, _dot(na, nb)))) < FEATURE_ANGLE:
                continue
            p = _add(foot, _mul(dc, lo))
            r = _add(foot, _mul(dc, hi))
            out.append((p, r, na, sa, nb, sb))
    return out


def find_edges(solid) -> list[Edge]:
    mesh = solid.mesh()
    v = mesh.vertices_flat()
    if not v:
        return []
    xs, ys, zs = v[0::3], v[1::3], v[2::3]
    scale = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1e-9)
    segs = _segments(mesh, scale)
    tol = 1e-6 * scale

    # merge collinear touching stretches with the same face pair -> straight edges
    def fkey(s):
        p, r, na, sa, nb, sb = s
        d = _unit(_sub(r, p))
        pair = sorted([tuple(round(c, 5) for c in na), tuple(round(c, 5) for c in nb)])
        return (tuple(round(c, 5) for c in d), tuple(pair))

    groups: dict = {}
    for s in segs:
        groups.setdefault(fkey(s), []).append(s)
    straight: list[Edge] = []
    for ss in groups.values():
        d = _unit(_sub(ss[0][1], ss[0][0]))
        ss.sort(key=lambda s: _dot(s[0], d))
        cur = None
        for s in ss:
            p, r = (s[0], s[1]) if _dot(_sub(s[1], s[0]), d) > 0 else (s[1], s[0])
            if cur and _len(_sub(p, cur[1])) < tol:
                cur[1] = r
            else:
                if cur:
                    straight.append(cur)
                cur = [p, r, s]
        if cur:
            straight.append(cur)
    edges = []
    for p, r, s in straight:
        _, _, na, sa, nb, sb = s
        edges.append(Edge("LINE", p, r, na, nb, sa, sb))

    # circles: chains of short straight edges that close in a plane
    short = [e for e in edges if e.length < 0.35 * scale]
    by_pt: dict = {}

    def pk(p):
        return tuple(round(c / tol) for c in p)

    for i, e in enumerate(short):
        by_pt.setdefault(pk(e.a), []).append(i)
        by_pt.setdefault(pk(e.b), []).append(i)
    used = set()
    circles = []
    for i0 in range(len(short)):
        if i0 in used:
            continue
        chain, pts = [i0], [short[i0].a, short[i0].b]
        used.add(i0)
        cur = short[i0].b
        ok = False
        for _ in range(len(short)):
            nxt = [j for j in by_pt.get(pk(cur), []) if j not in used]
            if not nxt:
                ok = _len(_sub(cur, pts[0])) < tol
                break
            j = nxt[0]
            used.add(j)
            chain.append(j)
            e = short[j]
            cur = e.b if _len(_sub(e.a, cur)) < tol else e.a
            pts.append(cur)
            if _len(_sub(cur, pts[0])) < tol:
                ok = True
                break
        if not ok or len(chain) < 12:
            continue
        ring = pts[:-1]
        c = _mul(tuple(sum(p[k] for p in ring) for k in range(3)), 1.0 / len(ring))
        rad = sum(_len(_sub(p, c)) for p in ring) / len(ring)
        if rad <= 0 or max(abs(_len(_sub(p, c)) - rad) for p in ring) > 1e-3 * rad:
            continue
        nrm = (0.0, 0.0, 0.0)
        for k in range(len(ring)):
            a, b = ring[k], ring[(k + 1) % len(ring)]
            nrm = _add(nrm, ((a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])))
        axis = _unit(nrm)
        e0 = short[chain[0]]
        circles.append((set(chain), Edge("CIRCLE", e0.midpoint, e0.midpoint, e0.n1, e0.n2, e0.t1, e0.t2,
                                         center=c, axis=axis, radius=rad, points=ring)))
    in_circle = set()
    for ch, _ in circles:
        in_circle |= {id(short[j]) for j in ch}
    return [e for e in edges if id(e) not in in_circle] + [c for _, c in circles]


# --------------------------------------------------------------- selection


def select(edges: list[Edge], rule) -> list[Edge]:
    """Filter ``edges`` by a compiled finder rule (a dict from the rule nodes)."""
    if rule is None:
        return edges
    k = rule.get("kind")
    if k == "and":
        out = edges
        for r in rule["rules"]:
            out = select(out, r)
        return out
    if k == "either":
        seen, out = set(), []
        for r in rule["rules"]:
            for e in select(edges, r):
                if id(e) not in seen:
                    seen.add(id(e))
                    out.append(e)
        return out
    if k == "parallelToAxis":
        ax = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}.get(str(rule["axis"]).upper())
        return [e for e in edges if ax and e.kind == "LINE" and abs(abs(_dot(e.direction, ax)) - 1) < 1e-6]
    if k == "inDirection":
        d = _unit(rule["direction"])
        return [e for e in edges if e.kind == "LINE" and abs(abs(_dot(e.direction, d)) - 1) < 1e-6]
    if k == "atAngle":
        d = _unit(rule["direction"])
        want = math.radians(rule["angle"])
        return [e for e in edges if e.kind == "LINE" and abs(math.acos(min(1.0, abs(_dot(e.direction, d)))) - min(want, math.pi - want)) < 1e-4]
    if k in ("isCircle", "ofType"):
        t = "CIRCLE" if k == "isCircle" else str(rule["type_name"]).upper()
        return [e for e in edges if e.kind == t]
    if k == "inPlane":
        o, n = rule["origin"], _unit(rule["normal"])
        o = _add(o, _mul(n, rule.get("distance", 0.0)))

        def on(p):
            return abs(_dot(_sub(p, o), n)) < 1e-6 * max(1.0, _len(p))

        return [e for e in edges if (on(e.a) and on(e.b)) if e.kind == "LINE"] + [
            e for e in edges if e.kind == "CIRCLE" and on(e.center) and abs(abs(_dot(e.axis, n)) - 1) < 1e-6
        ]
    if k == "containsPoint":
        p = rule["point"]
        out = []
        for e in edges:
            if e.kind == "LINE":
                d = e.direction
                s = _dot(_sub(p, e.a), d)
                foot = _add(e.a, _mul(d, s))
                if -1e-6 <= s <= e.length + 1e-6 and _len(_sub(p, foot)) < 1e-6 * max(1, e.length):
                    out.append(e)
            elif abs(_len(_sub(p, e.center)) - e.radius) < 1e-4 * e.radius and abs(_dot(_sub(p, e.center), e.axis)) < 1e-4 * e.radius:
                out.append(e)
        return out
    if k == "atDistance":
        p, dist = rule["point"], rule["distance"]

        def dline(e):
            d = e.direction
            s = max(0.0, min(e.length, _dot(_sub(p, e.a), d)))
            return _len(_sub(p, _add(e.a, _mul(d, s))))

        return [e for e in edges if e.kind == "LINE" and abs(dline(e) - dist) < 1e-6 * max(1, dist)]
    if k == "inBox":
        lo = tuple(min(a, b) for a, b in zip(rule["corner1"], rule["corner2"]))
        hi = tuple(max(a, b) for a, b in zip(rule["corner1"], rule["corner2"]))
        inside = lambda p: all(lo[i] - 1e-9 <= p[i] <= hi[i] + 1e-9 for i in range(3))
        return [e for e in edges if inside(e.center if e.kind == "CIRCLE" else e.midpoint)]
    if k == "lengthRange":
        return [e for e in edges if rule["min"] <= e.length <= rule["max"]]
    if k == "radiusRange":
        return [e for e in edges if e.kind == "CIRCLE" and rule["min"] <= e.radius <= rule["max"]]
    if k == "sortBy":
        inner = select(edges, rule["rule"])
        by = str(rule["by"]).upper()

        def key(e):
            m = e.center if e.kind == "CIRCLE" else e.midpoint
            if by in ("X", "Y", "Z"):
                return m["XYZ".index(by)]
            if by == "LENGTH":
                return e.length
            if by == "RADIUS":
                return e.radius
            return 0.0

        inner = sorted(inner, key=key)
        take = str(rule.get("take", "FIRST")).upper()
        if take == "FIRST":
            return inner[:1]
        if take == "LAST":
            return inner[-1:]
        return inner[: max(1, int(rule.get("n", 1)))]
    if k == "onFace":
        return edges  # the OCCT kernel does not resolve this rule either
    raise NotImplementedError(f"edge rule {k!r}")


# ----------------------------------------------------------------- tools


def _section(E, t1, t2, size, mode, size2=None):
    """2D section (3D points in the plane through E perpendicular to the edge)
    of the material a blend removes. ``mode`` = "fillet" | "chamfer"."""
    cos_th = max(-1.0, min(1.0, _dot(t1, t2)))
    th = math.acos(cos_th)  # interior angle between the faces
    bis = _unit(_add(t1, t2))
    eps = 1e-3 * size
    out = _sub(E, _mul(bis, eps))  # poke outside so the cut is clean
    if mode == "chamfer":
        d2 = size2 if size2 else size
        return [out, _add(E, _mul(t1, size)), _add(E, _mul(t2, d2))]
    k = size / math.tan(th / 2)
    T1, T2 = _add(E, _mul(t1, k)), _add(E, _mul(t2, k))
    C = _add(E, _mul(bis, size / math.sin(th / 2)))
    a1, a2 = _unit(_sub(T1, C)), _unit(_sub(T2, C))
    sweep = math.pi - th
    n = max(2, int(math.ceil(sweep / ANGULAR_STEP)))
    # arc from T1 to T2 around C, the short way (towards E)
    w = _unit(_cross(a1, a2))
    pts = [out, T1]
    for i in range(1, n):
        ang = sweep * i / n
        r = _add(_mul(a1, math.cos(ang)), _mul(_cross(w, a1), math.sin(ang)))
        pts.append(_add(C, _mul(r, size)))
    pts.append(T2)
    return pts


def _prism(pts, vec):
    wire = ist.make_polygon([ist.Pnt(*p) for p in pts])
    return ist.make_prism(ist.make_face(wire), ist.Pnt(*vec))


def _revolve(pts, center, axis):
    from .loft import _rotate

    n = 64
    rings = []
    for i in range(n + 1):
        ang = 2 * math.pi * i / n
        rings.append([_add(center, _rotate(_sub(p, center), axis, ang)) for p in pts])
    # a closed revolution: skin consecutive rings as a torus-like loop
    verts, tris = [], []
    m = len(pts)
    for ring in rings[:-1]:
        verts.extend(ring)
    for i in range(n):
        i2 = (i + 1) % n
        for j in range(m):
            j2 = (j + 1) % m
            a, b, c, d = i * m + j, i * m + j2, i2 * m + j2, i2 * m + j
            tris += [(a, b, c), (a, c, d)]
    vol = 0.0
    for a, b, c in tris:
        vol += _dot(verts[a], _cross(verts[b], verts[c]))
    if vol < 0:
        tris = [(a, c, b) for a, b, c in tris]
    return ist.solid_from_mesh([c for v in verts for c in v], [i for t in tris for i in t])


@dataclass
class BlendResult:
    solid: object
    blended: int = 0
    skipped: list = field(default_factory=list)


def blend(solid, edges: list[Edge], size: float, mode: str, size2: float | None = None) -> BlendResult:
    tools, skipped = [], []
    for e in edges:
        if not e.convex:
            skipped.append((e.kind, "concave"))
            continue
        if e.kind == "LINE":
            d = e.direction
            ext = 1e-3 * size
            A = _sub(e.a, _mul(d, ext))
            sec = _section(A, e.t1, e.t2, size, mode, size2)
            tools.append(_prism(sec, _mul(d, e.length + 2 * ext)))
        else:
            sec = _section(e.a, e.t1, e.t2, size, mode, size2)
            tools.append(_revolve(sec, e.center, e.axis))
    if not tools:
        return BlendResult(solid, 0, skipped)
    # one cut per tool: neighbouring tools overlap at shared corners with
    # nearly coplanar faces, which a single union of all tools mis-classifies
    for tool in tools:
        solid = ist.cut(solid, tool)
    return BlendResult(solid, len(tools), skipped)
