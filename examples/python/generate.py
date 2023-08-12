import cffi
import subprocess
import re
import os
from sys import argv
from pathlib import Path

c_compiler = os.environ.get('CC') or 'gcc'

def eval_size_expr(expr: str, header_files: list[Path], *args: list[str]) -> int:
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

import re
from sys import argv

def to_type(t):
  t = t.strip()
  if t.endswith('*'):
    return 'ffi.CData'
  elif t.startswith('struct '):
    return 'ffi.CData'
  elif t.startswith('enum ') or t in ['int', 'int64_t', 'size_t', 'int32_t', 'int16_t', 'int8_t', 'uint64_t', 'uint32_t', 'uint16_t', 'uint8_t']:
    return 'int'
  elif t in ['float', 'double']:
    return 'float'
  elif t == 'void':
    return 'None'
  elif t == 'bool':
    return 'bool'
  print('## Unknown type:', t)
  return None

def format_arg(t, n):
  t = to_type(t)
  return n if t is None else f'{n}: {t}'

def format_ret(t):
  t = to_type(t)
  return '' if t is None else f' -> {t}'

def generate_stubs(in_files: list[Path]):
    """
      Generates a .pyi Python stub file for the GGML API using C header files.

      The idea is to remove comments, then unbreak functions declared across multiple lines,
      the pseudo-parse the function name, return type, args and their types. Simple yet efficient.
    """
    original_header = '\n'.join([read_text(f) for f in in_files])
    header = original_header

    # Remove comments and ~ensure each GGML_API function is on a single line
    header = re.sub(r'/\*.*?\*/', '', header, flags=re.M | re.S)
    header = re.sub(r'//.*([\n\r])', '\\1', header)
    header = re.sub('([(,])\\s*[\\n\\r]\\s*', r'\1', header, flags=re.M | re.S)
    header = re.sub(r'[ \t]+', ' ', header)
    header = re.sub(r'\(\s*void\s*\)', '()', header)

    apis = list(set([l.strip() for l in header.splitlines() if 'GGML_API' in l and 'GGML_DEPRECATED' not in l]))
    apis.sort()

    lib = ['class lib:']
    for api in apis:
        m = re.search(r'GGML_API\s*(.*?)\b(\w+)\s*\(([^)]*)\)\s*;', api)
        if m is not None:
            try:
                (rettype, name, arglist) = m.groups()
                arglist = [a.strip() for a in arglist.split(',')]
                
                args = []
                for arg in arglist:
                    if arg.strip() == '':
                        continue
                    if arg == '...':
                        args.append('*args')
                    else:
                        am = re.search(r'^(.+?)\b(\w+)$', arg)
                        args.append(format_arg(am.group(1), am.group(2)))
                  
                lib += [
                    f'  # {api}', 
                    f'  def {name}({", ".join(args)}){format_ret(rettype)}: ...',
                    ''
                ]
            except:
                print('## Error parsing:', api)
                raise
    return '\n'.join(lib)

def prettify_header(header_files: list[Path], defines: list[str]) -> str:
    header = '\n'.join([read_text(f) for f in header_files])

    # Run preprocessor w/o any #includes (assume self-contained header, avoids noise from system defines)
    header = "\n".join([l for l in header.splitlines() if not re.match(r'^\s*#\s*include', l)])
    header = preprocess(header, *defines + [
        '-Ddeprecated(x)=',
        '-D__attribute__(x)=',
        '-Dstatic_assert(x, m)=',
    ])
    
    # Find all sizeof exprs and simple array bracket expressions constant and replace them w/ values computed by adhoc executables (because why not)
    constexpr_re = f'(?<=\\[)[^\\]]+(?=])|sizeof\\s*\\([^()]+\\)'
    constexprs = set([
       s for s in [
          m.group(0).strip()
          for m in re.finditer(constexpr_re, header)
       ] 
       if s != '' and not re.match(r'^\d+$', s)
    ])
    evals = {
       e: eval_size_expr(e, header_files, *defines)
       for e in constexprs
    }
    print(f'constexpr evals: {evals}')
    assert('sizeof(struct ggml_tensor)' in evals)

    for k, v in evals.items():
      header = header.replace(k, str(v))

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
    header_files = [
        include_dir / 'ggml.h',
        # include_dir / 'ggml-alloc.h',
        # include_dir / 'ggml-mpi.h',
    ]
    source_files = [
        src_dir / 'k_quants.c',
        src_dir / 'ggml.c',
    ]
else:
    include_dir = this_dir / '..' / '..' / 'include' / 'ggml'
    src_dir = this_dir / '..' / '..' / 'src'
    lib_dir = this_dir / '..' / '..' / 'build'
    header_files = [
        include_dir / 'ggml.h',
    ]
    source_files = [
        src_dir / 'ggml.c',
    ]

defines = [
    '-DGGML_USE_K_QUANTS',
]

header = prettify_header(header_files, defines)
# pycparser doesn't support __fp16, so replace it with uint16_t
# *but* don't do this with macros.
header = 'typedef uint16_t __fp16;\n' + header

if debug:
    opt_flags = ['-g', '-O0']
else:
    opt_flags = ['-Ofast', '-DNDEBUG']

ffibuilder = cffi.FFI()
ffibuilder.set_source(
    "ggml.cffi",
    "\n".join(list(map(read_text, source_files))) if compile else None,
    extra_link_args=["-lm"],
    extra_compile_args=opt_flags + defines +
      (["-I", include_dir.as_posix()] if compile else []),
)
try:
  ffibuilder.cdef(header)
  ffibuilder.compile(verbose=True)
except:
  print(header)
  raise

write_text(Path('ggml') / '__init__.pyi', f'''#
# AUTOGENERATED FILE, DO NOT EDIT
# TO REGENERATE, RUN:
#
#     python3 generate_stubs.py {' '.join(argv[1:])}
#
import ggml.ffi2 as ffi

# ggml library
{generate_stubs(header_files)}
''')