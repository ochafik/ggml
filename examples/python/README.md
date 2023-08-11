# Example usage of cffi-generated GGML bindings in Python

## Prerequisites

```bash
pip install -r requirements.txt

export LLAMA_DIR=$PWD/../../../llama.cpp
```

## Compile self-contained native extension (PREFERRED)

```bash
rm -fR pyggml/*

# Default env options: DEBUG=0 COMPILE=1 USE_LLAMA=1
python generate.py
# ls pyggml/*
#     ggml.c
#     ggml.o
#     ggml.cpython-310-darwin.so

# All good, just run this:
python example.py
```

## Point to libllama.so directly (no extension)

Alternatively you can load the compiled `libllama.so` binary w/ generated ctypes:

```bash
# Note: use LLAMA_DEBUG=1 to debug any crashes
( cd $LLAMA_DIR && LLAMA_METAL=1 make clean libllama.so )

rm -fR pyggml/*
COMPILE=0 python generate.py
# ls pyggml/*
#     ggml.py

# You can omit these if you've installed the library to say, /usr/lib
export DYLD_LIBRARY_PATH=$LLAMA_DIR:$DYLD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LLAMA_DIR:$LD_LIBRARY_PATH

python example.py
```
