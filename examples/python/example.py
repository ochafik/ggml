from ggml import ffi, lib

def set_floats_1d(t, values):
    for i, f in enumerate(values):
        lib.ggml_set_f32_1d(t, i, f)

def get_floats_1d(t):
    n = lib.ggml_nelements(t)
    return [lib.ggml_get_f32_1d(t, i) for i in range(n)]

if __name__ == '__main__':
    n_threads = 4
    n = 10

    params = ffi.new('struct ggml_init_params*')
    params.mem_size = 120000
    ctx = lib.ggml_init(params[0])

    a = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_F32, n)
    b = lib.ggml_new_tensor_1d(ctx, lib.GGML_TYPE_F32, n)

    set_floats_1d(a, [i for i in range(n)])
    set_floats_1d(b, [i + 100000 for i in range(n)])

    c = lib.ggml_add(ctx, a, b)
    c = lib.ggml_mul(ctx, c, c)

    gf = ffi.new('struct ggml_cgraph*')
    lib.ggml_build_forward_expand(gf, c)
    lib.ggml_graph_compute_with_ctx(ctx, gf, n_threads)

    print(get_floats_1d(a))
    print(get_floats_1d(b))
    print(get_floats_1d(c))
    # # lib.ggml_get_data(c)

    lib.ggml_free(ctx)