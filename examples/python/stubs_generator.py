"""
  This generates bindings for the ggml library using cffi and .pyi stubs for the Python bindings.

  See the various environment variables at the top of this file for options.
"""
import sys

# This is not required if you've installed pycparser into
# your site-packages/ with setup.py
sys.path.extend(['.', '..'])

import itertools
import re

from pycparser import c_ast, parse_file, CParser
import pycparser.plyparser
from pycparser.c_ast import PtrDecl, TypeDecl, FuncDecl, EllipsisParam, IdentifierType, Struct, Enum

__c_type_to_python_type = {
    'void': 'None', '_Bool': 'bool',
    'char': 'int', 'short': 'int', 'int': 'int', 'long': 'int',
    'ptrdiff_t': 'int', 'size_t': 'int',
    'int8_t': 'int', 'uint8_t': 'int',
    'int16_t': 'int', 'uint16_t': 'int',
    'int32_t': 'int', 'uint32_t': 'int',
    'int64_t': 'int', 'uint64_t': 'int',
    'float': 'float', 'double': 'float',
    'ggml_fp16_t': 'np.float16',
}

def format_type(t: TypeDecl):
    if isinstance(t, PtrDecl) or isinstance(t, Struct):
        return 'ffi.CData'
    if isinstance(t, Enum):
        return 'int'
    if isinstance(t, TypeDecl):
        return format_type(t.type)
    if isinstance(t, IdentifierType):
        assert len(t.names) == 1, f'Expected a single name, got {t.names}'
        name = t.names[0]
        return __c_type_to_python_type.get(name) or 'ffi.CData'
    return t.name

class PythonStubFuncDeclVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.sigs = {}
        self.sources = {}

    def get_source_snippet_lines(self, coord: pycparser.plyparser.Coord) -> list[str]:
        if coord.file not in self.sources:
            with open(coord.file, 'rt') as f:
                self.sources[coord.file] = f.readlines()
        source_lines = self.sources[coord.file]
        ncomments = len(list(itertools.takewhile(lambda i: re.search(r'^\s*(//|/\*)', source_lines[i]), range(coord.line - 2, -1, -1))))
        lines = []
        for line in source_lines[coord.line - 1 - ncomments:]:
            lines.append(line.rstrip())
            if (';' in line) or ('{' in line): break
        return lines

    def visit_Enum(self, node: Enum):
        if node.values is not None:
          for e in node.values.enumerators:
              self.sigs[e.name] = f'  @property\n  def {e.name}(self) -> int: ...'

    def visit_FuncDecl(self, node: FuncDecl):
        tp = node.type
        is_ptr = False
        while isinstance(tp, PtrDecl):
            tp = tp.type
            is_ptr = True
        
        name = tp.declname
        if name.startswith('__'):
            return
        
        args = []
        for a in node.args.params:
            if isinstance(a, EllipsisParam):
                args.append('*args')
            elif a.name is not None:
                args.append(f'{a.name}: {format_type(a.type)}')
        ret = format_type(tp if not is_ptr else node.type)

        self.sigs[name] = '\n'.join([
            *[f'  # {l}' for l in self.get_source_snippet_lines(node.coord)],
            f'  def {name}({", ".join(args)}) -> {ret}: ...',
        ])

def generate_stubs(header: str):
    """
      Generates a .pyi Python stub file for the GGML API using C header files.
    """

    with open('stubs.h', 'wt') as f:
        f.write(header)

    v = PythonStubFuncDeclVisitor()
    v.visit(CParser().parse(header, "<input>"))

    keys = list(v.sigs.keys())
    keys.sort()

    return '\n'.join(['class lib:', *[v.sigs[k] for k in keys]])
