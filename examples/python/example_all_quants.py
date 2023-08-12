from ggml import ffi, lib
from ggml.utils import init, numpy, copy
import numpy as np

if __name__ == '__main__':
    n_threads = 4
    
    ctx = init(mem_size = 1200000)
    
    def diff(a, b):
      return {
          "cos_sim": np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)),
          "linf": np.linalg.norm(b - a, np.inf),
          "l2": np.linalg.norm(b - a, 2)
      }

    n = 10 * 256

    # np.random.seed(1)
    a = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_F32, n)
    copy(np.random.rand(n).astype(np.float32), a)

    for type in range(lib.GGML_TYPE_COUNT):
        name = lib.ggml_type_name(type)
        name = ffi.string(name).decode('utf-8') if name else None

        if lib.ggml_is_quantized(type):
            # print(f'Testing quantization {name}')
            try:
                q = lib.ggml_new_tensor_1d(ctx, type, n)
                copy(a, q)
                d = diff(numpy(q, allow_copy=True), numpy(a))
                print(f'{name}: {d}')
            except Exception as e:
                print(f'Error: {e}')
