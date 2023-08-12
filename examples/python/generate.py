"""
  This generates bindings for the ggml library using cffi and .pyi stubs for the Python bindings.

  See --help for options.
  
  Usage:

    - `python generate.py --mode=dynamic_load --all_headers` to generate a Python module that can be versioned and supports all the headers
    - `python generate.py --mode=static_link --metal` on mac
    
"""

import cffi
import subprocess
import re
import os
from sys import argv
from pathlib import Path
from stubs_generator import generate_stubs
import platform
from typing import Literal, Optional
from jsonargparse import CLI


# C compiler to use (both for preprocessing and to evaluate some constant expressions)
c_compiler = os.environ.get('CC') or 'gcc'

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

def __read_text(p: Path):
   with p.open("rt") as f:
      return f.read()

def __write_text(p: Path, s: str):
  with p.open('wt') as f:
    return f.write(s)

def prettify_header(header_files: list[Path], defines: list[str]) -> str:
    header = '\n'.join([__read_text(f) for f in header_files])

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

def find_file_by_name(names: str, dir: Path, recurse=False) -> list[Path]:
    for name in names:
        candidates = [p for p in (dir.rglob if recurse else dir.glob)(name)]
        assert len(candidates) <= 1, f'Found multiple files matching {name} in {dir}: {candidates}'
        if len(candidates) == 1:
            return candidates[0]
    raise FileNotFoundError(f'Could not find any of {names} in {dir}')

def get_list_flag(n):
    v = os.environ.get(n)
    return [] if v is None else v.split(' ')

def generate(
        mode: Literal['dynamic_load', 'static_link', 'dynamic_link', 'inline'] = 'dynamic_load',
        llama_dir: Path = Path('.') / '..' / '..' / '..' / 'llama.cpp',
        ggml_dir: Path = Path('.') / '..' / '..',
        library: str = 'llama',
        ggml_package: str = 'ggml',
        build_dir: Path = None,
        all_headers: Optional[bool] = None,
        use_llama_cpp: bool = True,
        k_quants: Optional[bool] = None,
        accelerate: bool = platform.system() == 'Darwin',
        debug: bool = False,
        metal: bool = False,
        cuda: bool = False,
        opencl: bool = False,
        alloc: bool = True):
    
    CFLAGS = get_list_flag('CFLAGS')
    LDFLAGS = get_list_flag('LDFLAGS')

    if all_headers is not None:
        metal = True
        cuda = True
        opencl = True
        alloc = True

    if use_llama_cpp:
        if not llama_dir.is_dir():
            raise Exception(f'llama.cpp not found. set --use_llama_cpp=false or point --llama_dir to the right location')

        if k_quants is None:
            k_quants = True

        include_dir = llama_dir
        src_dir = llama_dir
    else:
        assert not k_quants, 'ggml does not support k_quants yet (if it does, please update this check!)'

        include_dir = ggml_dir / 'include' / 'ggml'
        src_dir = ggml_dir / 'src'

    if mode in ['static_link', 'dynamic_link']:
        assert build_dir is not None, f'build_dir must be specified when using {mode}'
        assert build_dir.is_dir()
        
    units = ['ggml']
    if k_quants:
        units += ['k_quants']
        CFLAGS += ['-DGGML_USE_K_QUANTS']
    if alloc:
        units += ['ggml-alloc']
    if cuda:
        units += ['ggml-cuda']
    if opencl:
        units += ['ggml-opencl']
    if metal:
        units += ['ggml-metal']
        CFLAGS += ['-DGGML_USE_METAL']
    if accelerate:
        CFLAGS += ['-DGGML_USE_ACCELERATE']
    
    header_files = [find_file_by_name([f'{n}.h'], include_dir) for n in units]
    
    preprocessed_header = prettify_header(header_files, defines=[])

    # pycparser doesn't support __fp16, so replace it with uint16_t
    preprocessed_header = 'typedef uint16_t __fp16;\n' + preprocessed_header
    CFLAGS += ['-D__fp16=uint16_t']

    # Generate stubs for the Python side. We don't use the preprocessed headers here
    # as they already lost some information (e.g. GGML_API).
    __write_text(Path(ggml_package) / '__init__.pyi', '\n'.join([
        '#',
        '# AUTOGENERATED STUB FILE, DO NOT EDIT',
        '# TO REGENERATE, RUN:',
        '#',
        '#     python generate.py',
        '#',
        f'import {ggml_package}.ffi as ffi',
        '',
        '# ggml library (cffi bindings)',
        generate_stubs('\n'.join([__read_text(f) for f in header_files])),
    ]))

    ffibuilder = cffi.FFI()
    ffibuilder.cdef(preprocessed_header)

    SOURCE = "\n".join([f'#include "{h.name}"' for h in header_files])
    source_files = []
    header_files = []

    LDFLAGS += ["-lm"]
    CFLAGS += ["-I", include_dir.as_posix()]
    if debug:
        CFLAGS += ['-g', '-O0']
    else:
        CFLAGS += ['-Ofast', '-DNDEBUG']

    if mode == 'dynamic_load':
        # We'll only generate ggml/cffi.py and ffi.dlopen magic will happen at / import time.
        SOURCE = None
    elif mode == 'dynamic_link':
        # cd llama.cpp && ( cmake . -B build -DBUILD_SHARED_LIBS=1 && cd build && make )
        # export LD_LIBRARY_PATH=$PWD/../../../llama.cpp/build:$LD_LIBRARY_PATH
        # export DYLD_LIBRARY_PATH=$PWD/../../../llama.cpp/build:$LD_LIBRARY_PATH
        lib_file = find_file_by_name([f'lib{library}.dylib', f'lib{library}.so'], build_dir)
        lib_dir = lib_file.parent
        LDFLAGS += ['-l', library] # lib_file.name]
        LDFLAGS += ['-L', lib_dir.as_posix()]
    else:
        # Do our best to link with the same flags as llama.cpp would.
        if accelerate:
            LDFLAGS += ['-framework', 'Accelerate']
        if metal:
            LDFLAGS += [
                '-framework', 'Foundation',
                '-framework', 'Metal',
                '-framework', 'MetalKit',
                '-framework', 'MetalPerformanceShaders',
            ]
        if mode == 'inline':
            # Inline all the source files into the native extension module and build it.
            # Compilation may or may not work well, and it's CPU-only (can't inline the metal code),
            # but it has the advantage it has no external dependencies.
            source_files = [find_file_by_name([f'{n}.c', f'{n}.cpp'], src_dir) for n in units]

            SOURCE += "\n" + "\n".join(list(map(__read_text, source_files)))
        elif mode == 'static_link':
            object_files = [
                find_file_by_name([f'{n}.o', f'{n}.c.o', f'{n}.cpp.o'], build_dir, recurse=True)
                for n in units
            ]
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

    ffibuilder.set_source(
        f'{ggml_package}.cffi',
        SOURCE,
        extra_link_args=LDFLAGS,
        extra_compile_args=CFLAGS)
    try:
        ffibuilder.compile(verbose=True)
    except:
        #   print(preprocessed_header)
        raise

    bindings_file = Path(ggml_package) / 'cffi.py'
    binary_files = Path(ggml_package).glob('cffi.*.so')
    if mode == 'dynamic_load':
        __write_text(bindings_file, "\n".join([
            '#',
            '# AUTOGENERATED *SLOW* BINDINGS FILE (--mode=dynamic_load)',
            '#',
            '# This file is versioned and can be used to power Python interop',
            '# provided libllama.so can be found, but it is not the fastest mode supported.',
            '#',
            '# If you need to make so many calls to the ggml API that per-call overhead becomes',
            '# a concern, please regenerate the bindings as a native extension.',
            '# To do so, run *something* like:',
            '#',
            '#     python generate.py --mode=static_link ...',
            '#',
            '', 
            __read_text(bindings_file),
        ]))
        # Note that Python hard-crashes Colab if it finds the binary *and* the bindings file at the same time.
        for binary_file in binary_files:
            binary_file.unlink()
    else:
        if bindings_file.exists():
            bindings_file.unlink()

if __name__ == '__main__':
    CLI(generate)