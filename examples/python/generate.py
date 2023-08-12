"""
  This generates bindings for the ggml library using cffi and .pyi stubs for the Python bindings.

  See the various environment variables at the top of this file for options.
"""

import cffi
import subprocess
import re
import os
from sys import argv
from pathlib import Path
from stubs_generator import generate_stubs
import platform
from typing import Literal

this_dir = Path('.')

# C compiler to use (both for preprocessing and to evaluate some constant expressions)
c_compiler = os.environ.get('CC') or 'gcc'

# Whether to use the headers & implementation of the llama.cpp library instead of ggml.
# This is currently useful as the ggml repo seems to be lagging behind the llama.cpp repo.
# (for instance, k_quants are only available in llama.cpp for now)
use_llama = os.environ.get('USE_LLAMA', '1') == '1'

# Whether to have cffi compile a native extension. It's what they call “out-of-line” + "API mode".
# If set to `0`, we're in “out-of-line” + “ABI mode” and cffi will only generate a ggml/cffi.py file
# that contains serialized type information and a few helpers to load the shared library.
compile = os.environ.get('COMPILE', '1') == '1'

# Build the native extension with debug symbols and no optimizations.
debug = os.environ.get('DEBUG', '0') == '1'

# Where to find the llama.cpp repo if USE_LLAMA is set to `1`.
llama_dir = Path(os.environ.get('LLAMA_DIR', (this_dir / '..' / '..' / '..' / 'llama.cpp').as_posix()))


def eval_size_expr(expr: str, header_files: list[Path], *args: list[str]) -> str:
    """
      Evaluates a constant C++ size expression using the C compiler.

      (e.g. `eval_size_expr("sizeof(uint64_t) / 2")` returns '4'
    """
    try:
        includes = '\n'.join([f'#include "{f.as_posix()}"' for f in header_files])
        subprocess.run(
            [c_compiler, "-o", "eval_size_expr", *args, "-x", "c", "-"],
            input=f'''
              #include <stdio.h>
              {includes}
              int main() {{ printf("%lu", (size_t)({expr})); }}
            ''', text=True, check=True)
        return subprocess.run(["./eval_size_expr"], capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        print(f"Failed to eval constexpr {expr}: {e}")
        return expr

def preprocess(src: str, *args: list[str]):
    return subprocess.run([c_compiler, "-E", *args, "-"], input=src, capture_output=True, text=True, check=True).stdout

def read_text(p: Path):
   with p.open("rt") as f:
      return f.read()

def write_text(p: Path, s: str):
  with p.open('wt') as f:
    return f.write(s)

def remove_nones(xs): return [x for x in xs if x is not None]

def prettify_header(header_files: list[Path], defines: list[str]) -> str:
    header = '\n'.join([read_text(f) for f in header_files])

    # Run preprocessor w/o any #includes (assume self-contained header, avoids noise from system defines)
    # Also, remove exotic annotations / calls that pycparser / cffi don't like.
    header = "\n".join([l for l in header.splitlines() if not re.match(r'^\s*#\s*include', l)])
    header = preprocess(header, *defines + [
        '-Ddeprecated(x)=',
        '-D__attribute__(x)=',
        '-Dstatic_assert(x, m)=',
    ])
    
    # pycparser (which powers cffi) doesn't like constant size expressions, so we
    # replace them with their values as part of our preprocessing.
    #
    # This regexp finds anyting *inside* square brackets (matched with lookbehind & lookahead)
    # and anything that looks like a sizeof call.
    constexpr_re = f'(?<=\\[)[^\\]]+(?=])|sizeof\\s*\\([^()]+\\)'
    constexprs = {
       e: eval_size_expr(e, header_files, *defines)
       for e in set(re.findall(constexpr_re, header))
       if not re.match(r'^(\d+|\s*)$', e)
    }
    print(f'constexprs: {constexprs}')
    assert('sizeof(struct ggml_tensor)' in constexprs) # sanity check
    for expr, v in constexprs.items():
      header = header.replace(expr, v)

    return header

def find_file_by_name(names: str, dir: Path, required, recurse=False) -> list[Path]:
    for name in names:
        candidates = [p for p in (dir.rglob if recurse else dir.glob)(name)]
        assert len(candidates) <= 1, f'Found multiple files matching {name} in {dir}: {candidates}'
        if len(candidates) == 1:
            return candidates[0]
    assert not required, f'Could not find any of {names} in {dir}'

if use_llama:
    include_dir = llama_dir
    src_dir = llama_dir
    lib_dir = llama_dir
    objects_dir = llama_dir
else:
    include_dir = this_dir / '..' / '..' / 'include' / 'ggml'
    src_dir = this_dir / '..' / '..' / 'src'
    lib_dir = this_dir / '..' / '..'
    objects_dir = this_dir / '..' / '..'

required_units = [
    'ggml'
]
optionals_units = [
    'k_quants',
    # 'ggml-metal',
    # 'ggml-alloc',
    # 'ggml-opencl',
    # 'ggml-cuda',
]
all_units = [*[(True, n) for n in required_units], *[(False, n) for n in optionals_units]]

header_files = remove_nones([
    find_file_by_name([f'{n}.h'], include_dir, required)
    for required, n in all_units
])

preprocessed_header = prettify_header(header_files, defines=[])
# pycparser doesn't support __fp16, so replace it with uint16_t
# *but* don't do this with macros.
preprocessed_header = 'typedef uint16_t __fp16;\n' + preprocessed_header

# Generate stubs for the Python side.
raw_concatenated_headers = '\n'.join([read_text(f) for f in header_files])

write_text(Path('ggml') / '__init__.pyi', f'''#
# AUTOGENERATED STUB FILE, DO NOT EDIT
# TO REGENERATE, RUN:
#
#     python generate.py
#
import ggml.ffi as ffi

# ggml library (cffi bindings)
{generate_stubs(raw_concatenated_headers)}
''')

def get_list_flag(n):
    v = os.environ.get(n)
    return [] if v is None else v.split(' ')

CFLAGS = get_list_flag('CFLAGS')
LDFLAGS = get_list_flag('LDFLAGS')
LDFLAGS += ["-lm"]

CFLAGS += ["-I", include_dir.as_posix()]
if debug:
    CFLAGS += ['-g', '-O0']
else:
    CFLAGS += ['-Ofast', '-DNDEBUG']

ffibuilder = cffi.FFI()
ffibuilder.cdef(preprocessed_header)

Mode = Literal['dynamic_load', 'static_link', 'dynamic_link', 'inline']

mode: Mode = os.environ.get('MODE', 'inline')
LIBRARY = os.environ.get('LIBRARY', 'llama')
# # Inline all the source files into the CFFI module.

LINKED_OBJS=[]
SOURCE = "\n".join([f'#include "{h.name}"' for h in header_files])
source_files = []
header_files = []

CFLAGS += ['-D__fp16=uint16_t']

if mode == 'dynamic_load':
    # We'll only generate ggml/cffi.py and ffi.dlopen magic will happen at / import time.
    SOURCE = None
elif mode == 'inline':
    # Inline all the source files into the native extension module and build it.
    # Compilation may or may not work well, and it's CPU-only (can't inline the metal code),
    # but it has the advantage it has no external dependencies.
    source_files = remove_nones([
        find_file_by_name([f'{n}.c', f'{n}.cpp'], src_dir, required)
        for required, n in all_units
    ])
    if any(p.name == 'k_quants.c' for p in source_files):
        CFLAGS += ['-DGGML_USE_K_QUANTS'] # Only useful if inlining k_quants.c into native extension

    SOURCE += "\n" + "\n".join(list(map(read_text, source_files)))
else:
    # We'll either link dynamically or statically.
    # We import the header files from the generated extension in both cases.
    # source = '#include "ggml.h"'

    if mode == 'dynamic_link':
        # cd llama.cpp && ( cmake . -B build -DBUILD_SHARED_LIBS=1 && cd build && make )
        # export LD_LIBRARY_PATH=$PWD/../../../llama.cpp/build:$LD_LIBRARY_PATH
        # export DYLD_LIBRARY_PATH=$PWD/../../../llama.cpp/build:$LD_LIBRARY_PATH
        LDFLAGS += ['-l', LIBRARY]
        lib_file = find_file_by_name([f'lib{LIBRARY}.so', f'lib{LIBRARY}.dylib'], lib_dir, required=True, recurse=True)
        lib_dir = lib_file.parent
        LDFLAGS += ['-L', lib_dir.as_posix()]
    elif mode == 'static_link':
        if platform.system() == 'Darwin':
            LDFLAGS += [
                '-framework', 'Accelerate',
                '-framework', 'Foundation',
                '-framework', 'Metal',
                '-framework', 'MetalKit',
                '-framework', 'MetalPerformanceShaders',
            ]

        object_files = remove_nones([
            find_file_by_name([f'{n}.o', f'{n}.c.o', f'{n}.cpp.o'], objects_dir, required, recurse=True)
            for required, n in all_units
        ])
        LDFLAGS += [f.as_posix() for f in object_files]
    else:
        raise ValueError(f'Unknown mode: {mode}')

print(f"""
    Building ggml in mode={mode}
     
    CFLAGS={CFLAGS}
    LDFLAGS={LDFLAGS}
    source_files={source_files}
    header_files={header_files}
    SOURCE=[...{len(SOURCE or [])} chars]
""")
# Generate ggml/cffi.py or ggml/cffi.c
ffibuilder.set_source(
    "ggml.cffi",
    SOURCE,
    extra_link_args=LDFLAGS + LINKED_OBJS,
    extra_compile_args=CFLAGS)
try:
    ffibuilder.compile(verbose=True)
except:
    #   print(preprocessed_header)
    raise
