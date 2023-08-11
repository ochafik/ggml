import cffi
import subprocess
import re
import os
from pathlib import Path

c_compiler = os.environ.get('CC') or 'gcc'

def sizeof(type: str, header_files: list[Path], *args: list[str]) -> int:
    includes = '\n'.join([f'#include "{f.as_posix()}"' for f in header_files])
    subprocess.run(
        [c_compiler, "-o", "sizeof", "-x", "c", "-", *args],
        input=f'''
          #include <stdio.h>
          {includes}
          int main() {{ printf("%lu", sizeof({type})); }}
        ''', text=True)
    return int(subprocess.run(["./sizeof"], capture_output=True, text=True).stdout)

def preprocess(src: str, *args: list[str]):
    return subprocess.run([c_compiler, "-E", *args, "-"], input=src, capture_output=True, text=True).stdout

def read_text(p: Path):
  with p.open("rt") as f:
    return f.read()

def prettify_header(header_files: list[str], defines: list[str]) -> str:
  header = '\n'.join([read_text(f) for f in header_files])

  # Run preprocessor w/o any #includes (assume self-contained header, avoids noise from system defines)
  header = "\n".join([l for l in header.splitlines() if not re.match(r'^\s*#\s*include', l)])
  header = preprocess(header, *defines,
                      '-D__fp16=uint16_t',
                      '-Ddeprecated(x)=',
                      '-D__attribute__(x)=',
                      '-Dstatic_assert(x, m)=')
  
  # Find all sizeof calls and replace them w/ values computed by adhoc executables (because why not)
  sized = set([m.group(1) for m in re.finditer(r'sizeof\s*\(([^)]+)\)', header)])
  sizes = {n: sizeof(n, header_files, *defines) for n in sized}
  header = re.sub(r'sizeof\s*\(([^)]+)\)', lambda m: str(sizes[m.group(1)]), header)
  print(f'sizes: {sizes}')
  assert('struct ggml_tensor' in sizes)
  # assert('block_q2_K' in sizes)

  # Replace static size_t constants w/ macros and run preprocessor again to expand them
  header = re.sub(r'static\s+const\s+size_t\s+(\w+)\s*=\s*([0-9]+)\s*;', r'#define \1 \2', header)
  header = preprocess(header, *defines)

  return header

this_dir = Path('.')

use_llama = os.environ.get('USE_LLAMA', '1') == '1'
compile = os.environ.get('COMPILE', '1') == '1'
debug = os.environ.get('DEBUG', '0') == '1'
llama_dir = Path(os.environ.get('LLAMA_DIR', (this_dir / '..' / '..' / '..' / 'llama.cpp').as_posix()))

if use_llama:
  include_dir = llama_dir
  src_dir = llama_dir
  lib_dir = llama_dir
else:
  include_dir = this_dir / '..' / '..' / 'include' / 'ggml'
  src_dir = this_dir / '..' / '..' / 'src'
  lib_dir = this_dir / '..' / '..' / 'build'

header_files = [
    # include_dir / 'k_quants.h',
    include_dir / 'ggml.h',
]
source_files = [
    # src_dir / 'k_quants.c',
    src_dir / 'ggml.c',
]
defines = [
    # '-DGGML_USE_K_QUANTS',
]

header = prettify_header(header_files, defines)
# print(header)

opt_flags = ['-g', '-O0'] if debug else ['-Ofast', '-DNDEBUG']

ffibuilder = cffi.FFI()
ffibuilder.set_source(
    "ggml.cffi",
    "\n".join(list(map(read_text, source_files))) if compile else None,
    extra_link_args=["-lm"],
    extra_compile_args=opt_flags + defines +
      (["-I", include_dir.as_posix()] if compile else []),
)
ffibuilder.cdef(header)
ffibuilder.compile(verbose=True)
