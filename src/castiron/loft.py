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
    secs = [resample_closed(s, n) for s in sections]

    # loft axis: first section centroid -> last section centroid
    c0 = tuple(sum(p[k] for p in secs[0]) / n for k in range(3))
    c1 = tuple(sum(p[k] for p in secs[-1]) / n for k in range(3))
    axis = tuple(c1[k] - c0[k] for k in range(3))
    if sum(a * a for a in axis) < 1e-18:
        axis = _newell_normal(secs[0])  # closed/flat loft: use the loop plane

    # orient every section CCW about the axis so wall quads all face outward
    secs = [
        s if sum(_newell_normal(s)[k] * axis[k] for k in range(3)) >= 0 else s[::-1]
        for s in secs
    ]

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


def _frames_along_path(path: list[Vec3]):
    """A rotation-minimizing-ish frame (origin, x-axis, y-axis) per path point."""
    m = len(path)
    tangents = []
    for i in range(m):
        a = path[max(i - 1, 0)]
        b = path[min(i + 1, m - 1)]
        t = tuple(b[k] - a[k] for k in range(3))
        ln = math.sqrt(sum(c * c for c in t)) or 1.0
        tangents.append(tuple(c / ln for c in t))
    frames = []
    up = (0.0, 0.0, 1.0)
    for i, tan in enumerate(tangents):
        ref = up if abs(sum(tan[k] * up[k] for k in range(3))) < 0.95 else (1.0, 0.0, 0.0)
        x = _cross(ref, tan)
        x = _norm(x)
        y = _norm(_cross(tan, x))
        frames.append((path[i], x, y))
    return frames


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(v):
    ln = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / ln for c in v)


def sweep_solid(profile2d: list[tuple[float, float]], path: list[Vec3]):
    """Sweep a 2D profile along a 3D path, orienting it to the path tangent."""
    sections = []
    for origin, x, y in _frames_along_path(path):
        sections.append([
            tuple(origin[k] + px * x[k] + py * y[k] for k in range(3))
            for (px, py) in profile2d
        ])
    return loft_solid(sections)
