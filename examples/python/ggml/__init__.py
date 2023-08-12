"""
  Python bindings for the ggml library.

  Usage example:

      from ggml import lib, ffi
      from ggml.utils import init, copy, numpy
      import numpy as np

      ctx = init(mem_size=10*1024*1024)
      n = 1024
      n_threads = 4

      a = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_Q5_K, n)
      b = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_F32, n)
      sum = lib.ggml_add(ctx, a, b)

      gf = ffi.new('struct ggml_cgraph*')
      lib.ggml_build_forward_expand(gf, sum)

      copy(np.array([i for i in range(n)], np.float32), a)
      copy(np.array([i*100 for i in range(n)], np.float32), b)
      lib.ggml_graph_compute_with_ctx(ctx, gf, n_threads)

      print(numpy(sum, allow_copy=True))

  This module is just a convenient hub to import the cffi-generated
  wrappers, supporting the two ways they can be built.

  - If the native extension was generated with `python generate.py`, 
    we're in what cffi calls “out-of-line” + “API mode” (the fastest)
    and this module imports both `lib` & `ffi` from ./ggml/cffi.*.so

  - If only a Python module was generated using `COMPILE=0 python generate.py`,
    we're in what cffi “out-of-line” + “ABI mode” 
    and this module imports `ffi` from ./ggml/cffi.py and creates `lib`
    by loading the llama (or ggml) shared library. That library can be
    found from the usual LD_LIBRARY_PATH or DYLD_LIBRARY_PATH, or its 
    full path specified in the GGML_PY_SO environment variable.
    
  See https://cffi.readthedocs.io/en/latest/cdef.html for more on cffi.

  Additionally, we generate stubs for `ggml.lib` in `ggml/lib/__init__.pyi`
  that tooling like VS Code or mypy can make use of.

"""
# Import / export the lib and ffi objects for the "low-level" bindings.
########################################################################################################################

# If the following import fails, you haven't run `python generate.py` yet.
try:
    from ggml.cffi import ffi as ffi
except ImportError:
    raise ImportError("Couldn't find ggml bindings. Run `python generate.py` first, or check your PYTHONPATH.")

try:
    # Try to import the "out-of-line" (compiled extension) version of the library.
    from ggml.cffi import lib as _lib
except ImportError:
    import os
    # We've only got Python bindings, so we need to load the library manually.
    # If the following line fails, add the directory containing the .so to DYLD_LIBRARY_PATH (on Mac) or LD_LIBRARY_PATH.
    _lib = ffi.dlopen(os.environ.get("GGML_PY_SO") or "libllama.so")

# This is where all the functions, enums and constants are defined
lib = _lib

# This contains the cffi helpers such as new, cast, string, etc.
# https://cffi.readthedocs.io/en/latest/ref.html#ffi-interface
ffi = ffi
