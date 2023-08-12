from ggml import lib, ffi
from ggml.utils import init, copy, numpy
import numpy as np

ctx = init(mem_size=12*1024*1024)
n = 256
n_threads = 4

a = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_Q5_K, n)
b = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_F32, n) # right-hand side must be F32 for add op
sum = lib.ggml_add(ctx, a, b) # all zeroes for now. Will be quantized too!

gf = ffi.new('struct ggml_cgraph*')
lib.ggml_build_forward_expand(gf, sum)

copy(np.array([i for i in range(n)], np.float32), a)
copy(np.array([i*100 for i in range(n)], np.float32), b)

lib.ggml_graph_compute_with_ctx(ctx, gf, n_threads)

print(numpy(a, allow_copy=True))
#  0.    1.0439453   2.0878906   3.131836    4.1757812   5.2197266. ...
print(numpy(b))
#  0.  100.        200.        300.        400.        500.         ...
print(numpy(sum, allow_copy=True))
#  0.  105.4375    210.875     316.3125    421.75      527.1875     ...