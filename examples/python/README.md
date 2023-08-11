# Example usage of cffi-generated GGML bindings in Python

### Prerequisites

```bash
pip install -r requirements.txt

export LLAMA_DIR=$PWD/../../../llama.cpp
```

### Compile self-contained native extension (PREFERRED)

```bash
rm -fR ggml/cffi.*

# Default env options: DEBUG=0 COMPILE=1 USE_LLAMA=1
python generate.py
python generate_stubs.py ../../../llama.cpp/{k_quants,ggml*}.h
# ls ggml/*
#     cffi.c
#     cffi.o
#     cffi.cpython-310-darwin.so

# All good, just run this:
python example.py
```

### Point to libllama.so directly (no extension)

Alternatively you can load the compiled `libllama.so` binary w/ generated .py bindings:

```bash
# Note: use LLAMA_DEBUG=1 to debug any crashes
( cd $LLAMA_DIR && LLAMA_METAL=1 make clean libllama.so )

rm -fR ggml/cffi.*
COMPILE=0 python generate.py
# ls ggml/*
#     cffi.py

# You can omit these if you've installed the library to say, /usr/lib
export DYLD_LIBRARY_PATH=$LLAMA_DIR:$DYLD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LLAMA_DIR:$LD_LIBRARY_PATH

python example.py
```

### Alternatives

This simple example's primary goal is to showcase automatically generated & fast bindings.

- https://github.com/abetlen/ggml-python: these bindings seem to be hand-written and use [ctypes](https://docs.python.org/3/library/ctypes.html) (making its maintenance potentially error-prone and performance slower than compiled [cffi](https://cffi.readthedocs.io/) bindings). It has [high-quality API reference docs](https://ggml-python.readthedocs.io/en/latest/api-reference/#ggml.ggml), possibly making it a sounder choice for serious development.
  
- https://github.com/abetlen/llama-cpp-python: these expose the C++ `llama.cpp` interface, which this example cannot easily be extended to support (`cffi` only generates bindings of C libraries)

- [pybind11](https://github.com/pybind/pybind11) and [nanobind](https://github.com/wjakob/nanobind) are two alternatives to cffi that generate bindings for C++. Unfortunately none of them have an automatic generator so writing bindings is quite time-consuming.

### Caveats

While [cffi](https://cffi.readthedocs.io/) makes it trivial to keep up with any changes to the GGML API, it's using the pycparser package which seems a bit sensitive to exotic C syntaxes (the likes that can be found in Mac system headers, for instance), and it doesn't have its own C preprocessor. See [generate.py](./generate.py) to get a better idea of the what was needed to make this work.
