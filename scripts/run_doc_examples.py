#!/usr/bin/env python3
"""Run every CADbuildr foundation documentation example through castiron.

The documentation site's examples all live in the monorepo's ``foundation_ex``
package (tsjs/packages/others/data/src/foundation_ex). Each example calls
``show(...)`` under ``if __name__ == "__main__"``. This runner executes each
file as ``__main__`` in a subprocess with ``show`` patched to
``castiron.compile`` — so "the example passes" means the real example code
compiled to real geometry on IronStream.

Usage:
    python scripts/run_doc_examples.py [--examples-dir DIR] [--only NAME]
    python scripts/run_doc_examples.py --child FILE   # internal
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Path to the foundation_ex package (CADbuildr monorepo,
# tsjs/packages/others/data/src/foundation_ex). Override with --examples-dir
# or the FOUNDATION_EX_DIR environment variable.
DEFAULT_EXAMPLES_DIR = os.environ.get("FOUNDATION_EX_DIR", "foundation_ex")
SKIP_NAMES = {"__init__.py", "test_all_examples.py"}
TIMEOUT_S = 300


# --------------------------------------------------------------------------
# child: run one example with show() patched to castiron.compile
# --------------------------------------------------------------------------


def run_child(path: str) -> None:
    import runpy
    import tempfile

    # Run inside the foundation_ex package so relative imports work: derive
    # the dotted module name from the path (…/foundation_ex/assemblies/x.py
    # -> foundation_ex.assemblies.x). sys.path already has the package parent.
    p = Path(path).resolve()
    mod_name = None
    for parent in p.parents:
        if parent.name == "foundation_ex":
            rel = p.relative_to(parent.parent)
            mod_name = ".".join(rel.with_suffix("").parts)
            break

    import cadbuildr.foundation as foundation
    from castiron import compile as castiron_compile
    from castiron.compiler import NothingToBuildError

    out_dir = tempfile.mkdtemp(prefix="castiron_doc_")
    results: list[dict] = []

    def patched_show(obj, *args, **kwargs):
        try:
            m = castiron_compile(obj, format="stl", out_dir=out_dir)
            results.append({
                "parts": [{"name": p["name"], "volume": p["volume"]} for p in m.parts],
            })
        except NothingToBuildError as e:
            # sketch-only / empty designs have nothing to export — not an error
            results.append({"parts": [], "note": str(e)})
        return "castiron"

    foundation.show = patched_show
    # some modules import show via submodules too
    try:
        foundation.dag_utils.show = patched_show
    except AttributeError:
        pass

    if mod_name:
        runpy.run_module(mod_name, run_name="__main__")
    else:
        runpy.run_path(path, run_name="__main__")

    if not results:
        print(json.dumps({"status": "no_show", "shows": []}))
    else:
        print(json.dumps({"status": "ok", "shows": results}))


# --------------------------------------------------------------------------
# parent: enumerate + fan out
# --------------------------------------------------------------------------


def discover(examples_dir: Path) -> list[Path]:
    out = []
    for p in sorted(examples_dir.rglob("*.py")):
        if p.name in SKIP_NAMES or "__pycache__" in p.parts:
            continue
        out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples-dir", default=DEFAULT_EXAMPLES_DIR)
    ap.add_argument("--only", help="substring filter on example path")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        run_child(args.child)
        return 0

    examples_dir = Path(args.examples_dir)
    files = discover(examples_dir)
    if args.only:
        files = [f for f in files if args.only in str(f)]

    env = dict(os.environ)
    # examples do absolute imports like `from foundation_ex.parts...`
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(examples_dir.parent), env.get("PYTHONPATH", "")])
    )

    ok, no_show, failed = [], [], []
    for f in files:
        rel = str(f.relative_to(examples_dir))
        proc = subprocess.run(
            [sys.executable, __file__, "--child", str(f)],
            capture_output=True, text=True, env=env, timeout=TIMEOUT_S,
        )
        line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        if proc.returncode == 0 and line.startswith("{"):
            data = json.loads(line)
            if data["status"] == "ok":
                vols = [round(p["volume"]) for s in data["shows"] for p in s["parts"]]
                print(f"OK       {rel:55} shows={len(data['shows'])} vols={vols}")
                ok.append(rel)
            else:
                print(f"NO-SHOW  {rel:55} (module has no __main__ show)")
                no_show.append(rel)
        else:
            err = (proc.stderr or "").strip().splitlines()
            msg = err[-1][:100] if err else f"exit {proc.returncode}"
            print(f"FAIL     {rel:55} {msg}")
            failed.append((rel, msg))

    print(f"\n=== {len(ok)} ok, {len(no_show)} no-show, {len(failed)} failed"
          f" / {len(files)} total ===")
    for rel, msg in failed:
        print(f"  FAIL {rel}: {msg}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
