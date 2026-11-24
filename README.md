# castiron

Compile a [CADbuildr foundation](https://github.com/cadbuildr) design into
geometry by *casting* its DAG into [IronStream](https://github.com/cadbuildr/ironstream)
solids via the [`ironstream`](https://github.com/cadbuildr/ironstream-python)
Python binding.

`castiron` is the **end-to-end Python path**: foundation (Python) →
`compile()` → IronStream kernel (Rust, via the Python binding) → a file on disk.
It is the offline, in-process counterpart to foundation's `show()` — no broker,
no network, no kernel-api.

```python
from cadbuildr.foundation import Part, Sketch, Point, Line, Polygon, Extrusion
from castiron import compile

class Cube(Part):
    def __init__(self, size=20):
        s = Sketch(self.xy())
        p1, p2, p3, p4 = Point(s,0,0), Point(s,size,0), Point(s,size,size), Point(s,0,size)
        poly = Polygon([Line(p1,p2), Line(p2,p3), Line(p3,p4), Line(p4,p1)])
        self.add_operation(Extrusion(poly, size))

manifest = compile(Cube(20), format="stl")
print(manifest.files)   # ['.cadbuildr/build/<dag-hash>/cube.stl']
```

`compile()` accepts a foundation `Part`/`Assembly` (or an already-serialized DAG
dict from `foundation.dag_utils.show_dag`) and writes `stl`, `step`, or `json`
(glTF-like mesh). Output is content-addressed by DAG hash, so a repeated build is
a cache hit:

```
<out_dir>/<dag_hash>/<part_name>.<ext>
<out_dir>/<dag_hash>/manifest.json
```

## How it works

Two pieces (see `src/castiron/`):

1. **`dag.py`** — the DAG tool: parses foundation's `CompilerInputDAG`, inverts
   the numeric type table to type *names*, and topologically orders the nodes.
2. **`nodes.py`** — the conversion functions: one handler per foundation node
   type, each emitting IronStream binding calls (`make_polygon` → `make_face` →
   `make_prism`, `make_revol`, booleans, …). `placement.py` maps 2D sketch
   coordinates into 3D world space.

Adding a node type = writing one handler and registering it in `HANDLERS`.

## Status

**OCCT parity.** On the kernel_compare benchmark (every buildable design in the
CADbuildr corpus: stdlib catalog, Prusa i3, Sesame robot, LEGO, playground,
jewellery, …), castiron matches the OCCT kernel on **111 of 112** designs
(IoU ≥ 0.95, volume within 2 %, bbox within 1 %). It is about 6× faster than
OCCT (geometric mean of warm compile times) and uses ~25× less memory (median
peak RSS 37 MB vs 895 MB). The one exception is a lathed-ellipse ring: OCCT's own
mesh is 2 % under the exact volume, while castiron is within 0.3 % of exact.

Covered: parameters; points/lines/arcs/splines/3D points; closed shapes
(polygon, rectangle, square, hexagon, circle, ellipse, custom); sketches /
planes / frames (quaternion placement); `Extrusion` (with taper and holes),
`Hole`, `Lathe`, `Sphere`; `Mirror`, `Scale`, `Shell`, `Stitch`; `Loft` /
`SurfaceLoft` / `Sweep` / `MultiSectionSweep`; `Helix3D` threads; `Fillet` /
`Chamfer` with `EdgeFinder` rules; `LinearPattern` / `PolarPattern` / `MoveBody`
/ `BooleanOperation`; planar / extruded / revolved / ruled surfaces;
sheet-metal base + edge flange; `Assembly` (multi-part, frame-chained
placement); `SVGShape` and SVG `<text>` (with the `[svg]` extra).

How OCCT semantics are matched where it matters:

- **Sweep** places the profile on its sketch plane at the path start, corrects
  it orthogonal to the tangent and moves it by the exact helical motion (OCCT
  `genericSweep` with Frenet on a helix). A helix starts on `gp_Ax2`'s automatic
  X axis like `makeHelix`. Paths built from `Point3D` are mapped by their part's
  frame.
- **Solid modifiers** (`Fillet`, `Chamfer`, `Shell`) replace the part's running
  solid, as in the OCCT kernel's `PartRoot`.
- **Tessellation** follows OCCT's angular step (64 segments per full circle,
  arcs in proportion to their sweep). The kernel re-merges coplanar boolean
  fragments, so the median triangle count is ~1.4× OCCT's.

Approximations / gaps, in order of fidelity cost:

- **Fillet / Chamfer** are geometric, because IronStream solids have no B-rep
  edges. Sharp edges are found on the mesh and a swept section is cut per
  convex straight or circular edge. Where filleted edges meet, the corner is the
  intersection of the roll-offs, not OCCT's spherical vertex patch (a fully
  rounded cube is within 1 % of exact). Concave edges, partial arcs and
  freeform edges are not blended. `OnFaceRule` is a no-op, as in the OCCT kernel.
- **Draft / FullRound** forward the solid unchanged.
- **Sheet-metal bends** (Tab/Hem/Miter/Unfold) forward the base flange; no bend
  deformation.
- **Face-selected features** (`OffsetFace`, `Emboss`, `Wrap`, `TextOnPath`,
  `Rip`, the sheet-metal face ops) and `PartModifier` have no handler. They
  need face identity, which a mesh solid does not carry. See `KNOWN_GAPS` in
  `tests/test_compile.py`.
- **Threads** come out at about 5× OCCT's triangle count: helical facets are
  fragmented by the shank's facets in the mesh boolean.

## Development

Install alongside the sibling repos (`ironstream-python`, `cadbuildr-foundation`):

```bash
uv pip install -e .
pytest
```
