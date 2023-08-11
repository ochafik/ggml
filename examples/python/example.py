from ggml import GgmlContext, GgmlTensor, lib, type_to_dtype
import numpy as np

if __name__ == '__main__':
    n_threads = 4
    n = 1024

    ctx = GgmlContext(mem_size=80000000)

    # TODO: support quantization
    # type = lib.GGML_TYPE_I8
    # type = lib.GGML_TYPE_Q4_0
    # type = lib.GGML_TYPE_Q4_K
    type = lib.GGML_TYPE_F16
    # type = lib.GGML_TYPE_F32
    dtype = type_to_dtype[type]

    a = ctx.new_tensor(type, n, n)
    b = ctx.new_tensor(type, n, n)
    # print(f'a: shape={a.shape}, strides={a.strides}')
    # print(f'b: shape={b.shape}, strides={b.strides}')

    # a.write([(i) for i in range(n * n)])
    a.write(np.ones((n, n), dtype=dtype), quantize=True)
    # b.write([(i * 2) for i in range(n * n)])
    b.write(np.random.rand(n, n).astype(dtype), quantize=True)
    # b.write(np.ones((n, n), dtype=dtype))

    # c = a * b
    c = a + b
    # c = c * c

    # d = a @ b

    gf = ctx.build_forward(c)
    # gf.build_forward_expand(d)

    gf.compute(n_threads)

    print(a.numpy())
    print(b.numpy())
    print(c.numpy()) # dequantize=True
