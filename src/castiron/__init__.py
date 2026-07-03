"""castiron — compile a CADbuildr foundation DAG into geometry via IronStream.

    from castiron import compile
    manifest = compile(my_part, format="stl")
    print(manifest.files)
"""

from .compiler import BuildManifest, Compiler, compile
from .dag import Dag, DagNode
from .placement import Placement

__all__ = ["compile", "Compiler", "BuildManifest", "Dag", "DagNode", "Placement"]
__version__ = "0.1.0"
