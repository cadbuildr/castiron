"""DXF (2D) export for castiron.

STL and STEP describe solids; DXF describes the 2D *drawing* — sketch profiles
today, and sheet-metal flat patterns in time. This writer harvests the 2D shape
nodes a design computed (closed shapes, analytic circles, standalone curves) and
emits a minimal AC1015 (AutoCAD 2000) DXF that FreeCAD, LibreCAD, Inkscape and
the usual laser-cutter toolchains all read.

Geometry is emitted in the *sketch's local 2D coordinates* — which is exactly
what a flat drawing is. A design drawn across several planes flattens into one
space; that's a known limitation until per-sketch layers land.
"""
from __future__ import annotations

from typing import Any

# Node types whose computed value is a closed 2D loop of (x, y) points.
_CLOSED = {"Polygon", "Rectangle", "Square", "Hexagon", "CustomClosedShape"}
# Node types whose computed value is an open 2D polyline the user drew as a
# standalone shape (as opposed to a Line/Arc that's a constituent of a shape).
_OPEN = {"CustomOpenShape", "Spline", "Bezier", "BSpline", "EllipseArc",
         "Offset2D", "Trace"}
# Nodes whose value is a Loops (many loops): SVG art, rendered text.
_MULTI = {"SVGShape", "Text"}


def _f(v: float) -> str:
    return f"{float(v):.6f}"


class Dxf:
    """Accumulates DXF entities and renders a complete AC1015 document."""

    def __init__(self) -> None:
        self._e: list[str] = []
        self.count = 0

    def polyline(self, pts, closed: bool) -> None:
        pts = [p for p in pts]
        if len(pts) < 2:
            return
        self._e += ["0", "LWPOLYLINE", "8", "0", "90", str(len(pts)),
                    "70", "1" if closed else "0", "43", "0"]
        for x, y in pts:
            self._e += ["10", _f(x), "20", _f(y)]
        self.count += 1

    def circle(self, center, radius: float) -> None:
        if radius <= 0:
            return
        self._e += ["0", "CIRCLE", "8", "0",
                    "10", _f(center[0]), "20", _f(center[1]), "30", "0.0",
                    "40", _f(radius)]
        self.count += 1

    def line(self, p1, p2) -> None:
        self._e += ["0", "LINE", "8", "0",
                    "10", _f(p1[0]), "20", _f(p1[1]), "30", "0.0",
                    "11", _f(p2[0]), "21", _f(p2[1]), "31", "0.0"]
        self.count += 1

    def render(self) -> str:
        head = ["0", "SECTION", "2", "HEADER",
                "9", "$ACADVER", "1", "AC1015",
                "9", "$INSUNITS", "70", "4",        # 4 = millimeters
                "0", "ENDSEC",
                "0", "SECTION", "2", "ENTITIES"]
        tail = ["0", "ENDSEC", "0", "EOF"]
        return "\n".join(head + self._e + tail) + "\n"


def build_dxf(compiler: Any) -> Dxf:
    """Harvest the 2D shapes a design explicitly drew into a DXF document.

    Emits the user-created shapes — closed loops, analytic circles, open curves,
    SVG/text loops — and deliberately skips their constituent ``Line``/``Arc``
    primitives, so a square isn't drawn on top of its own four edges and the
    "useless closing line" the pencil examples leave behind stays out of the cut
    file.
    """
    dag = compiler.dag
    store = compiler.store
    doc = Dxf()

    for nid, node in dag.nodes.items():
        t = node.type_name
        v = store.get(nid)
        if t == "Circle":
            c = store.get(node.deps.get("center"))
            r = store.get(node.deps.get("radius"))
            if c is not None and r:
                doc.circle(c, float(r))
        elif t == "Ellipse" and isinstance(v, list):
            doc.polyline(v, closed=True)
        elif t in _CLOSED and isinstance(v, list) and v:
            doc.polyline(v, closed=True)
        elif t in _OPEN and isinstance(v, list) and v:
            doc.polyline(v, closed=False)
        elif t in _MULTI and hasattr(v, "loops"):
            for loop in v.loops:
                doc.polyline(loop, closed=True)

    return doc
