"""Load upstream modWorm's Python routines without starting its Julia runtime.

Only module-level Julia definition registrations are omitted, using an AST
loader. Original files remain byte-identical. Calling a Julia routine raises.
The upstream BSD license is distributed with the pinned source snapshot.
"""
import ast
import importlib.abc
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys
import types


class _NoJulia:
    def __getattr__(self, name):
        raise RuntimeError('Julia is disabled: only verified Python routines are supported')


class _PythonLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        tree = ast.parse(Path(self.path).read_text(), filename=self.path)
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == 'Main' and node.value.func.attr == 'eval')]
        return compile(tree, self.path, 'exec')


class _PythonFinder(importlib.abc.MetaPathFinder):
    def __init__(self, package): self.package = package
    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith('modWorm.'): return None
        source = self.package / (fullname.split('.')[-1]+'.py')
        if not source.exists(): return None
        return importlib.util.spec_from_loader(fullname, _PythonLoader(fullname, str(source)))


def load_python_backend(upstream):
    upstream = Path(upstream).resolve()
    if 'modWorm' in sys.modules: raise RuntimeError('one upstream backend per process')
    package = types.ModuleType('modWorm')
    package.__path__ = [str(upstream/'modWorm')]
    package.Main = _NoJulia()
    sys.modules['modWorm'] = package
    sys.meta_path.insert(0, _PythonFinder(upstream/'modWorm'))
    before = Path.cwd()
    try:
        os.chdir(upstream)
        from modWorm import predefined_classes_nv, predefined_classes_mb, proprioception_simulation
    finally:
        os.chdir(before)
    return predefined_classes_nv, predefined_classes_mb, proprioception_simulation
