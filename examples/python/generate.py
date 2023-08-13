# Generates bindings for the ggml library using cffi
#
# pycparser (used by cffi) chokes on various C constructs found in system headers or in the ggml headers,
# so we help it a bit (e.g. replace sizeof expressions with their value, remove exotic syntax).
import os, sys, re, subprocess
import cffi

API = os.environ.get('API', 'api.h')
INCLUDE = os.environ.get('INCLUDE', '../../../llama.cpp')
CC = os.environ.get('CC') or 'gcc'
CPPFLAGS = [
    "-I", INCLUDE,
    '-D__fp16=uint16_t',  # pycparser doesn't support __fp16
    '-D__attribute__(x)=',
    '-D_Static_assert(x, m)=',
] + [x for x in os.environ.get('CPPFLAGS', '').split(' ') if x != '']

try: header = subprocess.run([CC, "-E", *CPPFLAGS, API], capture_output=True, text=True, check=True).stdout
except subprocess.CalledProcessError as e: print(f'{e.stderr}\n{e}', file=sys.stderr); raise

header = '\n'.join([l for l in header.split('\n') if '__darwin_va_list' not in l]) # pycparser hates this

# Replace constant size expressions w/ their value (compile & run a mini exe for each, because why not).
# First, extract anyting *inside* square brackets and anything that looks like a sizeof call.
for expr in set(re.findall(f'(?<=\\[)[^\\]]+(?=])|sizeof\\s*\\([^()]+\\)', header)):
    if re.match(r'^(\d+|\s*)$', expr): continue # skip constants and empty brackets    
    subprocess.run([CC, "-o", "eval_size_expr", *CPPFLAGS, "-x", "c", "-"], text=True, check=True,
                   input=f'''#include <stdio.h>
                             #include "{API}"
                             int main() {{ printf("%lu", (size_t)({expr})); }}''')
    size = subprocess.run(["./eval_size_expr"], capture_output=True, text=True, check=True).stdout
    print(f'constexpr {expr} = {size}')
    header = header.replace(expr, size)

ffibuilder = cffi.FFI()
ffibuilder.cdef(header)
ffibuilder.set_source(f'ggml.cffi', None)
ffibuilder.compile(verbose=True)
