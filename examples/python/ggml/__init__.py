########################################################################################################################
# Import / export the lib and ffi objects for the "low-level" bindings.
########################################################################################################################

# If the following import fails, you haven't run `python generate.py` yet.
from ggml.cffi import ffi as ffi

try:
  # Try to import the "out-of-line" (compiled extension) version of the library.
  from ggml.cffi import lib as _lib
except ImportError:
  # We've only got Python bindings, so we need to load the library manually.
  # If the following line fails, add the directory containing the .so to DYLD_LIBRARY_PATH (on Mac) or LD_LIBRARY_PATH.
  _lib = ffi.dlopen("libllama.so")

# This is where all the functions, enums and constants are defined
lib = _lib

# This contains the cffi helpers such as new, cast, string, etc.
# https://cffi.readthedocs.io/en/latest/ref.html#ffi-interface
ffi = ffi
