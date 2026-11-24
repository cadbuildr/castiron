"""Loft / sweep skinning, built on top of the IronStream binding.

The IronStream kernel does not expose a loft/sweep primitive, so — like the Rust
`ironstream-dag` reference — we assemble the mesh here (resample sections to a
common vertex count, build side quads, cap the ends) and hand the triangle soup
to ``ironstream.solid_from_mesh``.
"""

from __future__ import annotations

import math

import ironstream as ist

Vec3 = tuple[float, float, float]


def _dist(a: Vec3, b: Vec3) -> float:
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in range(3)))


def resample_closed(loop: list[Vec3], n: int) -> list[Vec3]:
    """Resample a closed 3D loop to exactly ``n`` points by arc length."""
    m = len(loop)
    seg = [_dist(loop[i], loop[(i + 1) % m]) for i in range(m)]
    total = sum(seg)
    if total == 0:
        return [loop[0]] * n
    acc = [0.0]
    for s in seg:
        acc.append(acc[-1] + s)
    out: list[Vec3] = []
    j = 0
    for k in range(n):
        t = total * k / n
        while j < m and acc[j + 1] < t:
            j += 1
        j = min(j, m - 1)
        f = (t - acc[j]) / (seg[j] or 1e-12)
        a, b = loop[j], loop[(j + 1) % m]
        out.append(tuple(a[c] + (b[c] - a[c]) * f for c in range(3)))
    return out


def upsample_closed(loop: list[Vec3], n: int) -> list[Vec3]:
    """Grow a closed loop to ``n`` points by repeatedly splitting its longest
    edge at the midpoint. Original vertices are kept in order."""
    pts = list(loop)
    while len(pts) < n:
        m = len(pts)
        k = max(range(m), key=lambda i: _dist(pts[i], pts[(i + 1) % m]))
        a, b = pts[k], pts[(k + 1) % m]
        pts.insert(k + 1, tuple((a[c] + b[c]) * 0.5 for c in range(3)))
    return pts


def _newell_normal(loop: list[Vec3]) -> Vec3:
    """Loop normal via Newell's method (robust for non-planar polygons)."""
    nx = ny = nz = 0.0
    m = len(loop)
    for i in range(m):
        x1, y1, z1 = loop[i]
        x2, y2, z2 = loop[(i + 1) % m]
        nx += (y1 - y2) * (z1 + z2)
        ny += (z1 - z2) * (x1 + x2)
        nz += (x1 - x2) * (y1 + y2)
    return (nx, ny, nz)


def _signed_volume(verts: list[Vec3], tris: list[tuple[int, int, int]]) -> float:
    """Signed volume by the divergence theorem (positive = outward winding)."""
    v = 0.0
    for a, b, c in tris:
        ax, ay, az = verts[a]
        bx, by, bz = verts[b]
        cx, cy, cz = verts[c]
        v += (ax * (by * cz - bz * cy)
              + ay * (bz * cx - bx * cz)
              + az * (bx * cy - by * cx))
    return v / 6.0


def loft_solid(sections: list[list[Vec3]]):
    """Skin a closed solid through an ordered list of 3D section loops.

    Winding is made globally consistent and outward: every section is oriented
    counter-clockwise about the loft axis before the walls are built, the end
    caps follow that orientation, and a final signed-volume check flips the
    whole mesh if needed — so ironstream, trimesh and any STL consumer agree.
    """
    sections = [s for s in sections if len(s) >= 3]
    if len(sections) < 2:
        raise ValueError("loft needs at least two sections")
    n = max(len(s) for s in sections)
    # Keep every section's own vertices (corners of a polygonal profile carry
    # the shape); only sections with fewer points get extra points, inserted
    # on their longest edges. Arc-length resampling of all sections cut the
    # corners off a 4-point thread profile and shrank it ~18x.
    secs = [s if len(s) == n else upsample_closed(s, n) for s in sections]

    # Orient each section CCW about its *local* loft direction (towards the
    # next section), so wall quads face outward even when the spine curls
    # back on itself (a helix of non-integer turns, a U-bend). A reversed
    # section keeps its first vertex, so vertex j still pairs with vertex j
    # of its neighbours.
    cents = [tuple(sum(p[k] for p in s) / len(s) for k in range(3)) for s in secs]
    oriented = []
    for i, sec in enumerate(secs):
        a_, b_ = (cents[i], cents[i + 1]) if i + 1 < len(secs) else (cents[i - 1], cents[i])
        d = tuple(b_[k] - a_[k] for k in range(3))
        if sum(x * x for x in d) < 1e-18:
            d = _newell_normal(secs[0])  # coincident centroids: use the loop plane
        if sum(_newell_normal(sec)[k] * d[k] for k in range(3)) < 0:
            sec = [sec[0]] + sec[:0:-1]
        oriented.append(sec)
    secs = oriented

    verts: list[Vec3] = []
    starts: list[int] = []
    for s in secs:
        starts.append(len(verts))
        verts.extend(s)
    tris: list[tuple[int, int, int]] = []

    # side walls — with CCW sections about the axis, (v00,v11,v10)/(v00,v01,v11)
    # is the outward winding (cross of ascent × CCW tangent points outward)
    for a in range(len(secs) - 1):
        ia, ib = starts[a], starts[a + 1]
        for j in range(n):
            j2 = (j + 1) % n
            v00, v01, v10, v11 = ia + j, ia + j2, ib + j, ib + j2
            tris.append((v00, v11, v10))
            tris.append((v00, v01, v11))

    # end caps (fan from each end loop's centroid); with CCW sections the
    # bottom cap must face -axis (reversed fan) and the top cap +axis
    def cap(start: int, pts: list[Vec3], flip: bool):
        c = tuple(sum(p[k] for p in pts) / len(pts) for k in range(3))
        ci = len(verts)
        verts.append(c)
        for j in range(n):
            j2 = (j + 1) % n
            a, b = start + j, start + j2
            tris.append((ci, b, a) if flip else (ci, a, b))

    cap(starts[0], secs[0], flip=True)
    cap(starts[-1], secs[-1], flip=False)

    # belt and braces: if the closed mesh still encloses negative volume,
    # flip everything (also catches a reversed axis choice)
    if _signed_volume(verts, tris) < 0:
        tris = [(a, c, b) for a, b, c in tris]

    vflat = [c for v in verts for c in v]
    tflat = [i for t in tris for i in t]
    return ist.solid_from_mesh(vflat, tflat)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(v):
    ln = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / ln for c in v)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _rotate(v, axis, ang):
    """Rodrigues: rotate vector ``v`` about unit ``axis`` by ``ang`` radians."""
    c, s = math.cos(ang), math.sin(ang)
    kxv = _cross(axis, v)
    kdv = _dot(axis, v)
    return tuple(v[i] * c + kxv[i] * s + axis[i] * kdv * (1 - c) for i in range(3))


def _any_perp(v):
    ref = (1.0, 0.0, 0.0) if abs(v[0]) < 0.9 else (0.0, 1.0, 0.0)
    return _cross(v, ref)


def _min_rotation(a, b):
    """(axis, angle) of the smallest rotation taking unit ``a`` onto unit ``b``."""
    ax = _cross(a, b)
    sn = math.sqrt(_dot(ax, ax))
    cs = max(-1.0, min(1.0, _dot(a, b)))
    if sn < 1e-12:
        return None if cs > 0 else (_norm(_any_perp(a)), math.pi)
    return (_mul(ax, 1.0 / sn), math.atan2(sn, cs))


def ax2_xdir(v):
    """The X direction OCCT's ``gp_Ax2(P, V)`` picks for main direction ``V``,
    so a helix built on ``gp_Ax3(center, dir)`` starts where OCCT's does."""
    a, b, c = v
    aa, ba, ca = abs(a), abs(b), abs(c)
    if ba <= aa and ba <= ca:
        x = (-c, 0.0, a) if aa > ca else (c, 0.0, -a)
    elif aa <= ba and aa <= ca:
        x = (0.0, -c, b) if ba > ca else (0.0, c, -b)
    else:
        x = (-b, a, 0.0) if aa > ba else (b, -a, 0.0)
    return _norm(x)


class HelixPath(list):
    """Sampled helix points that also remember the analytic helix, so a Sweep
    can move its profile by the exact helical motion (what OCCT's Frenet law
    reduces to on a helix) instead of frames estimated from the samples."""

    def __init__(self, pts, center, axis, radius, pitch, height, sign, turns):
        super().__init__(pts)
        self.center, self.axis, self.radius = center, axis, radius
        self.pitch, self.height, self.sign, self.turns = pitch, height, sign, turns

    def tangent0(self):
        e1 = _norm(_sub(self[0], self.center))
        e2 = _cross(self.axis, e1)
        rise = self.pitch if self.turns else self.height
        return _norm(_add(_mul(e2, self.sign * 2 * math.pi * self.radius), _mul(self.axis, rise)))


def _rmf(path):
    """Rotation-minimising frames (double reflection, Wang et al. 2008) along a
    polyline: [(tangent, r)] per point. OCCT's corrected-Frenet law is a
    rotation-minimising frame too."""
    m = len(path)
    tans = []
    for i in range(m):
        a = path[max(i - 1, 0)]
        b = path[min(i + 1, m - 1)]
        tans.append(_norm(_sub(b, a)))
    r = _norm(_any_perp(tans[0]))
    out = [(tans[0], r)]
    for i in range(m - 1):
        v1 = _sub(path[i + 1], path[i])
        c1 = _dot(v1, v1)
        if c1 < 1e-24:
            out.append((tans[i + 1], r))
            continue
        rl = _sub(r, _mul(v1, 2.0 / c1 * _dot(v1, r)))
        tl = _sub(tans[i], _mul(v1, 2.0 / c1 * _dot(v1, tans[i])))
        v2 = _sub(tans[i + 1], tl)
        c2 = _dot(v2, v2)
        r = rl if c2 < 1e-24 else _sub(rl, _mul(v2, 2.0 / c2 * _dot(v2, rl)))
        r = _norm(r)
        out.append((tans[i + 1], r))
    return out


def sweep_solid(profile2d, path, placement=None, twist=0.0):
    """Sweep a 2D sketch profile along a 3D path the way the OCCT kernel does
    (replicad ``genericSweep`` with ``forceProfileSpineOthogonality``):

    * the profile is drawn on its sketch plane's axes, re-origined at the
      path's start point;
    * it is rotated about that point so its plane is orthogonal to the path
      tangent (OCCT's ``WithCorrection``);
    * every section is the start section moved by the path's frame law: the
      exact helical motion for a ``HelixPath``, rotation-minimising frames for
      any other polyline.
    """
    from .placement import quat_rotate

    if placement is not None:
        q = placement.quaternion
        xd, nd = quat_rotate(q, (1.0, 0.0, 0.0)), quat_rotate(q, (0.0, 0.0, 1.0))
    else:
        xd, nd = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    yd = _cross(nd, xd)
    origin = tuple(path[0])
    base = [_add(origin, _add(_mul(xd, px), _mul(yd, py))) for (px, py) in profile2d]

    helix = isinstance(path, HelixPath)
    frames = None if helix else _rmf(path)
    t0 = path.tangent0() if helix else frames[0][0]
    n0 = nd if _dot(nd, t0) >= 0 else _mul(nd, -1.0)
    rot = _min_rotation(_norm(n0), t0)
    if rot is not None:
        axis, ang = rot
        base = [_add(origin, _rotate(_sub(p, origin), axis, ang)) for p in base]

    sections = []
    if helix:
        k = path.axis
        last = len(path) - 1
        for i in range(len(path)):
            f = i / last
            ang = path.sign * 2 * math.pi * path.turns * f
            dz = path.height * f
            tw = twist * f
            sec = []
            for p in base:
                if tw:
                    p = _add(origin, _rotate(_sub(p, origin), t0, tw))
                rel = _sub(p, path.center)
                sec.append(_add(path.center, _add(_rotate(rel, k, ang), _mul(k, dz))))
            sections.append(sec)
    else:
        t_0, r_0 = frames[0]
        b_0 = _cross(t_0, r_0)
        local = []
        for p in base:
            d = _sub(p, origin)
            local.append((_dot(d, r_0), _dot(d, b_0), _dot(d, t_0)))
        last = max(1, len(frames) - 1)
        for i, (t, r) in enumerate(frames):
            b = _cross(t, r)
            tw = twist * i / last
            c, s = math.cos(tw), math.sin(tw)
            sections.append([
                _add(tuple(path[i]), _add(_add(_mul(r, u * c - v * s), _mul(b, u * s + v * c)), _mul(t, w)))
                for (u, v, w) in local
            ])
    return loft_solid(sections)
