# Generates bindings for the ggml library.
#
# cffi requires prior C preprocessing of the headers, and it uses pycparser which chokes on a couple of things
# so we help it a bit (e.g. replace sizeof expressions with their value, remove exotic syntax found in Darwin headers).
import os, sys, re, subprocess
import cffi
from stubs import generate_stubs

API = os.environ.get('API', 'api.h')
CC = os.environ.get('CC', 'gcc')
C_INCLUDE_DIR = os.environ.get('C_INCLUDE_DIR', '../../../llama.cpp')
CPPFLAGS = [x for x in os.environ.get('CPPFLAGS', '').split(' ') if x != '']

try: header = subprocess.check_output([
    CC,
    '-I', C_INCLUDE_DIR,
    '-U__GNUC__',
    '-D_Nullable=',
    '-D__asm(x)=',
    '-D__attribute__(x)=',
    '-D_Static_assert(x, m)=',
    *CPPFLAGS,
    '-E',
    API,
], text=True)
except subprocess.CalledProcessError as e: print(f'{e.stderr}\n{e}', file=sys.stderr); raise

# Replace constant sizeof expressions w/ their value (compile & run a mini exe for each, because why not).
for expr in set(re.findall(f'sizeof\\s*\\([^()]+\\)', header)):
    subprocess.run(
        [CC, "-o", "eval_size_expr", '-I', C_INCLUDE_DIR, *CPPFLAGS, "-x", "c", "-"],
        text=True, check=True,
        input=f'''#include <stdio.h>
                  #include "{API}"
                  int main() {{ printf("%lu", (size_t)({expr})); }}''')
    size = subprocess.check_output(["./eval_size_expr"], text=True)
    print(f'Computed constexpr {expr} = {size}')
    header = header.replace(expr, size)

ffibuilder = cffi.FFI()
ffibuilder.cdef(header, override=True)
ffibuilder.set_source(f'ggml.cffi', None) # we're not compiling a native extension, as this quickly gets hairy
ffibuilder.compile(verbose=True)

with open("ggml/__init__.pyi", "wt") as f:
    f.write(generate_stubs(header))