"""Placing 2D sketch geometry into 3D world space.

A foundation ``Sketch`` lives on a ``Plane`` whose ``Frame`` carries a position
and a quaternion. Sketch-local coordinates ``(x, y)`` (with an out-of-plane
offset ``z`` used for extrusion depth) map to world space by rotating with the
quaternion and translating by the position.

The rotation is done here in pure Python so the IronStream binding only ever
receives world-space points — no quaternion constructor needed on the Rust side.
"""

from __future__ import annotations

from dataclasses import dataclass


def quat_rotate(q: tuple[float, float, float, float], v: tuple[float, float, float]):
    """Rotate vector ``v`` by quaternion ``q = (w, x, y, z)``."""
    w, x, y, z = q
    vx, vy, vz = v
    # t = 2 * cross(q_xyz, v);  v' = v + w*t + cross(q_xyz, t)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


@dataclass(frozen=True)
class Placement:
    """A rigid placement: position + orientation quaternion ``(w, x, y, z)``."""

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def to_world(self, x: float, y: float, z: float = 0.0) -> tuple[float, float, float]:
        rx, ry, rz = quat_rotate(self.quaternion, (x, y, z))
        px, py, pz = self.position
        return (rx + px, ry + py, rz + pz)

    def normal(self) -> tuple[float, float, float]:
        """World-space plane normal (local +Z rotated by the orientation)."""
        return quat_rotate(self.quaternion, (0.0, 0.0, 1.0))
