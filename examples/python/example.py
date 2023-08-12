from ggml import ffi, lib
from ggml.utils import numpy, copy, type_to_dtype
import numpy as np

def set_floats_1d(t, values):
    for i, f in enumerate(values):
        lib.ggml_set_f32_1d(t, i, f)

def get_floats_1d(t):
    n = lib.ggml_nelements(t)
    return [lib.ggml_get_f32_1d(t, i) for i in range(n)]

if __name__ == '__main__':
    n_threads = 4
    n = 256

    params = ffi.new('struct ggml_init_params*')
    params.mem_size = 12000000
    ctx = lib.ggml_init(params[0])

    # type = lib.GGML_TYPE_I8
    type = lib.GGML_TYPE_Q4_0
    # type = lib.GGML_TYPE_Q4_K
    # type = lib.GGML_TYPE_F16
    # type = lib.GGML_TYPE_F32
    dtype = type_to_dtype(type) or np.float32

    # qtype = lib.GGML_TYPE_Q4_0
    qtype = lib.GGML_TYPE_Q8_0

    # a = lib.ggml_new_tensor_1d(ctx, type, n)
    # b = lib.ggml_new_tensor_1d(ctx, type, n)
    a = lib.ggml_new_tensor_2d(ctx, lib.GGML_TYPE_Q4_0, n, n)
    aq1 = lib.ggml_new_tensor_2d(ctx, lib.GGML_TYPE_Q8_0, n, n)
    aq2 = lib.ggml_new_tensor_2d(ctx, lib.GGML_TYPE_Q6_K, n, n)
    b = lib.ggml_new_tensor_2d(ctx, lib.GGML_TYPE_F32, n, n)

    rr = lib.ggml_new_tensor_2d(ctx, lib.GGML_TYPE_Q4_0, n, n)
    copy(np.random.rand(n * n).astype(np.float32).reshape((n, n)), rr)
    print(f'rr = {numpy(rr, allow_copy=True)}')

    # copy(np.array([i for i in range(n)]).astype(dtype), a)
    # copy(np.array([i + 100000 for i in range(n)]).astype(dtype), b)

    copy(np.array([(i) for i in range(n * n)]).astype(dtype).reshape((n, n)), a)

    copy(a, aq1)
    copy(a, aq2)
    print("QUANTIZED!")
    print(numpy(a, allow_copy=True))
    print(numpy(aq1, allow_copy=True))
    print(numpy(aq2, allow_copy=True))

    copy(np.array([(i * 2) for i in range(n * n)]).astype(dtype).reshape((n, n)), b)
    # copy(np.ones((n, n), dtype=dtype), a)
    # copy(np.random.rand(n, n).astype(dtype), b)

    # copy(np.ones((n, n), dtype=dtype), b)

    c = lib.ggml_add(ctx, a, b)
    c = lib.ggml_add(ctx, aq1, c)
    c = lib.ggml_add(ctx, aq2, c)
    # c = lib.ggml_mul(ctx, c, c)
    c = lib.ggml_mul_mat(ctx, a, b)

    c = lib.ggml_mul(ctx, c, c)

    gf = ffi.new('struct ggml_cgraph*')
    lib.ggml_build_forward_expand(gf, c)
    lib.ggml_graph_compute_with_ctx(ctx, gf, n_threads)

    # print(ffi.typeof(ffi.new('float[]', 10)))
    print(numpy(a, allow_copy=True))
    print(numpy(b, allow_copy=True))
    print(numpy(c, allow_copy=True))

    # # lib.ggml_get_data(c)

    lib.ggml_free(ctx)