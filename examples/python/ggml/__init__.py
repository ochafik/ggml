"""
  Python bindings for the ggml library.

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
from ggml.cffi import ffi as ffi
import os

try:
  # Try to import the "out-of-line" (compiled extension) version of the library.
  from ggml.cffi import lib as _lib
except ImportError:
  # We've only got Python bindings, so we need to load the library manually.
  # If the following line fails, add the directory containing the .so to DYLD_LIBRARY_PATH (on Mac) or LD_LIBRARY_PATH.
  _lib = ffi.dlopen(os.environ.get("GGML_PY_SO") or "libllama.so")

# This is where all the functions, enums and constants are defined
lib = _lib

# This contains the cffi helpers such as new, cast, string, etc.
# https://cffi.readthedocs.io/en/latest/ref.html#ffi-interface
ffi = ffi
