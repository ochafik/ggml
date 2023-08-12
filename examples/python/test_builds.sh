#!/bin/bash
set -euo pipefail

LLAMA_DIR=${LLAMA_DIR:-../../../llama.cpp}
GGML_DIR=${GGML_DIR:-../..}

function announce() {
  echo "#"
  echo "# $@"
  echo "#"
}

function test() {
  local target="$1"
  local mode="$MODE"

  prefix="[-> $target] MODE=$mode"
  announce "$prefix"

  rm -f ggml/cffi.*
  if ! python generate.py ; then
    announce "$prefix generate.py FAILED" >&2
    exit 1
  fi
  # LIB_LLAMA_SO=../../../llama.cpp/libllama.so 
  if ! python example.py ; then
    announce "$prefix example.py FAILED" >&2
    exit 1
  fi
}

# Build llama.cpp
( cmake "$LLAMA_DIR" -B llama_build -DBUILD_SHARED_LIBS=1 && \
  cd llama_build && \
  make -j )

( MODE=dynamic_load \
  export LD_LIBRARY_PATH=$PWD/llama_build ; \
  export DYLD_LIBRARY_PATH=$PWD/llama_build ; \
  test llama.cpp )

MODE=static_link \
  test llama.cpp

( export LD_LIBRARY_PATH=$PWD/llama_build ; \
  export DYLD_LIBRARY_PATH=$PWD/llama_build ; \
  MODE=dynamic_load \
  test llama.cpp )

MODE=inline \
  test llama.cpp
