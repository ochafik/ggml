# Import / export the lib and ffi objects from the generated module (which may or may not be compiled as an extension)
#
# If this import fails, you haven't run `python generate.py` yet.
from ggml.cffi import ffi as _ffi

try:
  from ggml.cffi import lib as _lib
except ImportError:
  # If the .so load fails, add the directory containing the .so to DYLD_LIBRARY_PATH (on Mac) or LD_LIBRARY_PATH.
  lib = _ffi.dlopen("libllama.so")

# This is where all the functions, enums and constants are defined
lib = _lib

# This contains the cffi helpers such as new, cast, string, etc.
ffi = _ffi

