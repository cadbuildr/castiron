"""The castiron compiler: foundation DAG -> IronStream geometry -> file.

``compile(p)`` takes a foundation ``Part``/``Assembly`` (or an already
serialized DAG dict), walks the DAG, runs each node's conversion function
against the IronStream Python binding, and writes the result to disk.

Output is content-addressed by the DAG hash so a repeated build is a cache hit
and paths are deterministic::

    <out_dir>/<dag_hash>/<part_name>.<ext>
    <out_dir>/<dag_hash>/manifest.json
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import ironstream as ist

from .dag import Dag, DagNode
from .nodes import HANDLERS, OpResult, PartResult, UnsupportedNodeError

FORMATS = {"stl", "step", "json"}
_EXT = {"stl": "stl", "step": "step", "json": "json"}


@dataclass
class BuildManifest:
    root_type: str
    dag_hash: str
    format: str
    out_dir: str
    parts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def files(self) -> list[str]:
        return [p["file"] for p in self.parts]


class Compiler:
    """Walks a :class:`Dag`, resolving each node into a value in ``store``."""

    def __init__(self, dag: Dag):
        self.dag = dag
        self.store: dict[str, Any] = {}

    # dependency-resolution helpers used by node handlers
    def get(self, node: DagNode, key: str) -> Any:
        return self.store[node.deps[key]]

    def get_opt(self, node: DagNode, key: str, default: Any = None) -> Any:
        ref = node.deps.get(key)
        if isinstance(ref, str):
            return self.store[ref]
        return default

    def get_list(self, node: DagNode, key: str) -> list[Any]:
        return [self.store[r] for r in node.deps.get(key, [])]

    def run(self) -> "Compiler":
        for nid in self.dag.topo_order():
            node = self.dag.nodes[nid]
            handler = HANDLERS.get(node.type_name)
            if handler is None:
                raise UnsupportedNodeError(
                    f"no castiron handler for node type {node.type_name!r} "
                    f"(id {nid}). Add one in castiron/nodes.py."
                )
            self.store[nid] = handler(self, node)
        return self

    @property
    def result(self) -> Any:
        return self.store[self.dag.root_id]


# --- public API ------------------------------------------------------------


def _to_dag_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict) and "DAG" in obj:
        return obj
    # A foundation Part / Assembly (or *Root) -> serialize via foundation.
    from cadbuildr.foundation.dag_utils import show_dag

    return show_dag(obj)


def _dag_hash(dag_dict: dict[str, Any]) -> str:
    canonical = json.dumps(dag_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class NothingToBuildError(ValueError):
    pass


def _as_parts(result: Any) -> list[PartResult]:
    if result is None:
        return []
    if isinstance(result, PartResult):
        return [result]
    if isinstance(result, OpResult):
        return [PartResult("part", result.solid)]
    if isinstance(result, list):
        parts: list[PartResult] = []
        for r in result:
            parts.extend(_as_parts(r))
        return parts
    # e.g. a sketch-only design whose root is a Placement/loop — no solid.
    raise NothingToBuildError(
        f"root node compiled to {type(result).__name__}, not a part/solid "
        "(a sketch-only design has no geometry to export)"
    )


def _write_solid(solid: "ist.Solid", fmt: str, path: Path, name: str) -> None:
    mesh = solid.mesh()
    if fmt == "stl":
        path.write_bytes(ist.write_binary_stl(mesh))
    elif fmt == "step":
        path.write_text(ist.write_step(mesh, name))
    elif fmt == "json":
        path.write_text(json.dumps({
            "vertices": mesh.vertices_flat(),
            "triangles": mesh.triangles_flat(),
            "volume": mesh.volume(),
        }))
    else:  # pragma: no cover - guarded by caller
        raise ValueError(fmt)


def compile_meshes(obj: Any) -> list[dict[str, Any]]:
    """Compile a foundation part/assembly (or DAG) to in-memory mesh data.

    No file I/O — returns one dict per part with flat ``vertices`` /
    ``triangles`` buffers, ready for a renderer. This is what the browser
    playground uses (pyodide -> three.js).
    """
    dag_dict = _to_dag_dict(obj)
    dag = Dag.from_compiler_input(dag_dict)
    parts = _as_parts(Compiler(dag).run().result)
    if not parts:
        raise NothingToBuildError("design produced no solid geometry")
    out: list[dict[str, Any]] = []
    for part in parts:
        mesh = part.solid.mesh()
        out.append({
            "name": part.name,
            "vertices": mesh.vertices_flat(),
            "triangles": mesh.triangles_flat(),
            "volume": part.solid.volume(),
        })
    return out


def compile(
    obj: Any,
    format: str = "stl",
    out_dir: str = ".cadbuildr/build",
    name: str | None = None,
) -> BuildManifest:
    """Compile a foundation part/assembly (or DAG) to a file via IronStream.

    Args:
        obj: a foundation ``Part``/``Assembly`` (or ``*Root``), or a serialized
            DAG dict as produced by ``foundation.dag_utils.show_dag``.
        format: ``"stl"``, ``"step"`` or ``"json"`` (glTF-like mesh).
        out_dir: base output directory; results go under ``<out_dir>/<hash>/``.
        name: optional override for the single-part file name.

    Returns:
        A :class:`BuildManifest` describing the DAG hash and written files.
    """
    if format not in FORMATS:
        raise ValueError(f"unknown format {format!r}; expected one of {sorted(FORMATS)}")

    dag_dict = _to_dag_dict(obj)
    dag = Dag.from_compiler_input(dag_dict)
    compiler = Compiler(dag).run()
    parts = _as_parts(compiler.result)
    if not parts:
        raise NothingToBuildError("design produced no solid geometry to export")

    dag_hash = _dag_hash(dag_dict)
    base = Path(out_dir) / dag_hash[:16]
    base.mkdir(parents=True, exist_ok=True)
    ext = _EXT[format]

    manifest = BuildManifest(
        root_type=dag.nodes[dag.root_id].type_name,
        dag_hash=dag_hash,
        format=format,
        out_dir=str(base),
    )

    single = len(parts) == 1
    used: dict[str, int] = {}
    for i, part in enumerate(parts):
        pname = (name if single and name else part.name) or f"part_{i}"
        # disambiguate repeated part names so files don't overwrite each other
        if pname in used:
            used[pname] += 1
            pname = f"{pname}_{used[pname]}"
        else:
            used[pname] = 0
        fp = base / f"{pname}.{ext}"
        _write_solid(part.solid, format, fp, pname)
        manifest.parts.append({
            "name": pname,
            "file": str(fp),
            "volume": part.solid.volume(),
        })

    (base / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2))
    return manifest
