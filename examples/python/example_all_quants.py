from ggml import ffi, lib
from ggml.utils import init, numpy, copy
import numpy as np

def diff(a, b):
  return { "l2": np.linalg.norm(b - a, 2), "linf": np.linalg.norm(b - a, np.inf) }

ctx = init(mem_size = 10*1024*1024)

n = 256
a = lib.ggml_new_tensor_2d(ctx, lib.GGML_TYPE_F32, n, n)
copy(np.random.rand(n * n).reshape((n, n)).astype(np.float32), a)

for type in range(lib.GGML_TYPE_COUNT):
    name = lib.ggml_type_name(type)
    name = ffi.string(name).decode('utf-8') if name else None

    if lib.ggml_is_quantized(type):
        # print(f'Testing quantization {name}')
        try:
            q = lib.ggml_new_tensor_2d(ctx, type, n, n)
            copy(a, q)
            d = diff(numpy(q, allow_copy=True), numpy(a))
            print(f'{name}: {d}')
        except Exception as e:
            print(f'Error: {e}')
