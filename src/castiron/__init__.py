"""castiron — compile a CADbuildr foundation DAG into geometry via IronStream.

    from castiron import compile
    manifest = compile(my_part, format="stl")
    print(manifest.files)
"""

from .compiler import BuildManifest, Compiler, NothingToBuildError, compile, compile_meshes
from .dag import Dag, DagNode
from .placement import Placement

__all__ = [
    "compile", "compile_meshes", "Compiler", "BuildManifest",
    "NothingToBuildError", "Dag", "DagNode", "Placement",
]
__version__ = "0.1.0"
