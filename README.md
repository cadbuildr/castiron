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

Handlers exist for every foundation node type. Against the kernel-truck fixture
corpus, **64/70 compile to geometry** and 5 more are legitimately sketch-only or
empty designs (no solid) — 69/70 accounted for.

Covered: parameters; points/lines/arcs/splines/3D points; closed shapes
(polygon, rectangle, square, hexagon, circle, ellipse, custom); sketches /
planes / frames (quaternion placement); `Extrusion`, `Hole`, `Lathe`, `Sphere`;
`Mirror`, `Scale`, `Shell`, `Stitch`; `Loft` / `SurfaceLoft` / `Sweep` /
`MultiSectionSweep` / `Helix3D` (mesh skinning via `ironstream.solid_from_mesh`);
sheet-metal base + edge flange (other bends forward the flange, matching the Rust
reference); `Assembly` (multi-part, frame-chained placement); and `SVGShape`
(path flattening).

Approximations / gaps, in order of fidelity cost:

- **Fillet / Chamfer / Draft / FullRound** forward the solid unchanged — the
  IronStream kernel exposes no edge-blend op yet (neither does the Rust reference).
- **Sheet-metal bends** (Tab/Hem/Miter/Unfold) forward the base flange; no bend
  deformation.
- **SVGShape** flattens `<path>` data as a single loop (multi-subpath holes and
  self-intersecting art are approximate); SVG `<text>` needs a font engine and is
  unsupported.

## Development

Install alongside the sibling repos (`ironstream-python`, `cadbuildr-foundation`):

```bash
uv pip install -e .
pytest
```
