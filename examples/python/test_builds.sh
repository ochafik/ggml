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
  local prefix="${prefix:-}[$@]"
  announce "$prefix"

  rm -f ggml/cffi.*

  if ! python generate.py "$@" ; then
    announce "$prefix generate.py FAILED" >&2
    exit 1
  fi

  if ! python example.py "$@" ; then
    announce "$prefix example.py FAILED" >&2
    exit 1
  fi
}

function test_against_llama() {
  local cmake_flags="$1"
  local generate_flags="$2"

  prefix="$cmake_flags $generate_flags"

  BUILD_DIR=/tmp/llama_build_for_ggml_python_example
  rm -fR $BUILD_DIR

  # Build llama.cpp
  ( cmake "$LLAMA_DIR" -B "$BUILD_DIR" -DBUILD_SHARED_LIBS=1 && \
    cd "$BUILD_DIR" && \
    make -j )

  # Dynamic load = easiest & fastest to generate. Slowest when doing lots of API calls.
  ( export LD_LIBRARY_PATH=$BUILD_DIR ; \
    export DYLD_LIBRARY_PATH=$BUILD_DIR ; \
    test --mode=dynamic_load --llama_dir="$LLAMA_DIR" )

  # Static link: easiest to use locally but a bit iffy.
  test --mode=static_link --llama_dir="$LLAMA_DIR" "--build_dir=$BUILD_DIR"

  # Dynamic link: proably the best option for serious deployment.
  ( export LD_LIBRARY_PATH=$BUILD_DIR ; \
    export DYLD_LIBRARY_PATH=$BUILD_DIR ; \
    export GGML_LIBRARY=$BUILD_DIR/libllama.dylib ; \
    test --mode=dynamic_link --llama_dir="$LLAMA_DIR" "--build_dir=$BUILD_DIR" )

  # Inline ggml sources in extension: CPU-only!!
  test --mode=inline --llama_dir="$LLAMA_DIR"
}

function test_against_ggml() {
  local cmake_flags="$1"
  local generate_flags="$2"

  prefix="$cmake_flags $generate_flags"

  BUILD_DIR=/tmp/ggml_build_for_python_example
  rm -fR $BUILD_DIR

  # Build ggml
  ( cmake "$GGML_DIR" -B "$BUILD_DIR" -DBUILD_SHARED_LIBS=1 && \
    cd "$BUILD_DIR" && \
    make -j )

  # Static link: easiest to use locally but a bit iffy.
  TEST_K_QUANTS=0 test --mode=static_link --ggml_dir="$GGML_DIR" "--build_dir=$BUILD_DIR" --use_llama_cpp=false

  # Inline ggml sources in extension: CPU-only!!
  TEST_K_QUANTS=0 test --mode=inline --ggml_dir="$GGML_DIR" --use_llama_cpp=false
}

test_against_llama "" ""

if [[ -d /Applications/Xcode.app ]]; then
  test_against_llama "-DLLAMA_METAL" "--metal --accelerate"
fi

# test_against_llama "-DLLAMA_CUDA" "--cuda"

test_against_ggml "" ""
