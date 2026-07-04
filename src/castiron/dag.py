"""The DAG tool: parse and walk the DAG that CADbuildr foundation emits.

foundation serializes a ``Part`` / ``Assembly`` into a ``CompilerInputDAG``::

    {
      "version": "...",
      "rootNodeId": "<id>",
      "DAG": { "<id>": {"type": <int>, "params": {...}, "deps": {...}}, ... },
      "serializableNodes": {"<TypeName>": <int>, ...}
    }

Dependencies are references (by node id) to other nodes; ``deps`` values are
either a single id (string) or a list of ids. This module inverts the numeric
type table into type *names* — the compiler dispatches on names, never on the
per-DAG numeric ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DagNode:
    id: str
    type_name: str
    params: dict[str, Any] = field(default_factory=dict)
    deps: dict[str, Any] = field(default_factory=dict)


@dataclass
class Dag:
    root_id: str
    nodes: dict[str, DagNode]
    version: str = ""

    @classmethod
    def from_compiler_input(cls, data: dict[str, Any]) -> "Dag":
        type_by_id = {v: k for k, v in data["serializableNodes"].items()}
        nodes: dict[str, DagNode] = {}
        for nid, raw in data["DAG"].items():
            nodes[nid] = DagNode(
                id=nid,
                type_name=type_by_id[raw["type"]],
                params=raw.get("params") or {},
                deps=raw.get("deps") or {},
            )
        return cls(root_id=data["rootNodeId"], nodes=nodes, version=data.get("version", ""))

    def dep_ids(self, node: DagNode) -> list[str]:
        """Flattened list of node ids this node depends on."""
        out: list[str] = []
        for v in node.deps.values():
            if isinstance(v, str):
                out.append(v)
            elif isinstance(v, (list, tuple)):
                out.extend(x for x in v if isinstance(x, str))
        return out

    def topo_order(self) -> list[str]:
        """Ids reachable from the root, dependencies before dependents."""
        seen: set[str] = set()
        order: list[str] = []

        def visit(nid: str) -> None:
            if nid in seen:
                return
            seen.add(nid)
            for dep in self.dep_ids(self.nodes[nid]):
                visit(dep)
            order.append(nid)

        visit(self.root_id)
        return order
