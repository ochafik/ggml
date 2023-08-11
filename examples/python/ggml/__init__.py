########################################################################################################################
# Import / export the lib and ffi objects for the "low-level" bindings.
########################################################################################################################

# If the following import fails, you haven't run `python generate.py` yet.
from ggml.cffi import ffi as _ffi

try:
  # Try to import the "out-of-line" (compiled extension) version of the library.
  from ggml.cffi import lib as _lib
except ImportError:
  # We've only got Python bindings, so we need to load the library manually.
  # If the following line fails, add the directory containing the .so to DYLD_LIBRARY_PATH (on Mac) or LD_LIBRARY_PATH.
  lib = _ffi.dlopen("libllama.so")

# This is where all the functions, enums and constants are defined
lib = _lib

# This contains the cffi helpers such as new, cast, string, etc.
# https://cffi.readthedocs.io/en/latest/ref.html#ffi-interface
ffi = _ffi

########################################################################################################################
# Below is a lightweight Object-oriented API that brings interop w/ numpy and automatic cleanup of the context.
########################################################################################################################

import numpy as np
from typing import Optional, Union

type_to_dtype = {
  lib.GGML_TYPE_I8: np.int8,
  lib.GGML_TYPE_I16: np.int16,
  lib.GGML_TYPE_I32: np.int32,
  lib.GGML_TYPE_F16: np.float16,
  lib.GGML_TYPE_F32: np.float32,
}

class GgmlTensor:
  def __init__(self, ctx, type, ptr: ffi.CData):
    self.type = type
    self.ptr = ptr
    self.ctx = ctx
    self.is_int = type in [lib.GGML_TYPE_I8, lib.GGML_TYPE_I16, lib.GGML_TYPE_I32]
    
  @property
  def shape(self):
    return [self.ptr.ne[i] for i in range(self.ptr.n_dims)]
  
  def build_forward(self):
    return GgmlComputationGraph(self.ctx, lib.ggml_build_forward(self.ptr))

  def __add__(self, other): return self.ctx.add(self, other)
  def __sub__(self, other): return self.ctx.sub(self, other)
  def __mul__(self, other): return self.ctx.mul(self, other)
  def __div__(self, other): return self.ctx.div(self, other)
  def __truediv__(self, other): return self.ctx.div(self, other)
  def __neg__(self): return self.ctx.unary(self, lib.GGML_UNARY_OP_NEG)
  def __matmul__(self, other): return self.ctx.mul_mat(self, other)

  def reshape(self, shape_or_tensor):
    if isinstance(shape_or_tensor, GgmlTensor):
       return self.__op(lib.ggml_reshape, shape_or_tensor)
    else:
      shape = shape_or_tensor
      if len(shape) == 1:
        return self.__op(lib.ggml_reshape_1d, shape[0])
      elif len(shape) == 2:
        return self.__op(lib.ggml_reshape_2d, shape[0], shape[1])
      elif len(shape) == 3:
        return self.__op(lib.ggml_reshape_3d, shape[0], shape[1], shape[2])
      elif len(shape) == 4:
        return self.__op(lib.ggml_reshape_4d, shape[0], shape[1], shape[2], shape[3])
      else:
        raise NotImplementedError()

  @property
  def strides(self):
    return [self.ptr.nb[i] for i in range(self.ptr.n_dims)]

  def write(self, values, quantize=False):
    if isinstance(values, np.ndarray):
      if values.shape != tuple(self.shape):
        raise ValueError(f"Shape mismatch: tensor has {self.shape} but values has {values.shape}")
      
      data = lib.ggml_get_data(self.ptr)

      dtype = type_to_dtype.get(self.type)
      if not dtype and lib.ggml_is_quantized(self.type):
        if not quantize:
          raise ValueError("Writing to tensor requires quantization. Force with quantize=True")
        dtype = np.float32
        if dtype != values.dtype:
          raise ValueError(f"Can only quantize from float32 but values are {values.dtype}")

        ttraits = lib.ggml_internal_get_type_traits(self.type)
        # destination = dequantize if isinstance(dequantize, np.ndarray) else np.empty(tuple(shape), dtype=np.float32)
        # TODO: np.ascontiguousarray(destination)? Or assert that it's contiguous.
        ttraits.from_float(ffi.cast("float*", ffi.from_buffer(values)), data, values.size)
        
        return
      if dtype != values.dtype:
        raise ValueError(f"Value type mismatch: tensor has {dtype} but values has {values.dtype}")
      
      ffi.memmove(data, ffi.from_buffer(values), values.nbytes)
      return
    else:
      for i, f in enumerate(values):
        self[i] = f

  def numpy(self, dequantize: Union[bool, np.ndarray] = False) -> np.ndarray:
    shape = self.shape
    strides = self.strides
    nbytes = lib.ggml_nbytes(self.ptr)
    assert(nbytes == strides[-1] * shape[-1])

    data = lib.ggml_get_data(self.ptr)
    
    if lib.ggml_is_quantized(self.type):
      if dequantize == False:
        raise ValueError("Quantized tensor requires extra memory to be converted to numpy array, and changes to the numpy array aren't reflected back to the tensor. Force with dequantize=True")
      
      ttraits = lib.ggml_internal_get_type_traits(self.type)
      destination = dequantize if isinstance(dequantize, np.ndarray) else np.empty(tuple(shape), dtype=np.float32)
      # TODO: np.ascontiguousarray(destination)? Or assert that it's contiguous.
      ttraits.to_float(data, ffi.buffer(destination), lib.ggml_nelements(self.ptr))
      return destination
    else:
      dtype = type_to_dtype.get(self.type)
      if not dtype:
        raise NotImplementedError(f'Unknown type {self.type}')
    
    a = np.frombuffer(ffi.buffer(data, nbytes), dtype=dtype)
    a.shape = tuple(self.shape)
    return a

  def get_floats_1d(self):
    n = lib.ggml_nelements(t)
    return [self[i] for i in range(n)]

  def set_f32_1d(self, i, v): lib.ggml_set_f32_1d(self.ptr, i, v)
  def get_f32_1d(self, i): lib.ggml_get_f32_1d(self.ptr, i)

  def set_i32_1d(self, i, v): lib.ggml_set_i32_1d(self.ptr, i, v)
  def get_i32_1d(self, i): lib.ggml_get_i32_1d(self.ptr, i)

  def __getitem__(self, index):
    (self.get_i32_1d if self.is_int else self.get_f32_1d)(index)

  def __setitem__(self, index, value):
    (self.set_i32_1d if self.is_int else self.set_f32_1d)(index, value)

class GgmlContext:
  def __init__(self, mem_size=None, mem_buffer=ffi.NULL, no_alloc=False):
      params = ffi.new('struct ggml_init_params*')
      params.mem_size = mem_size
      params.mem_buffer = mem_buffer
      params.no_alloc = no_alloc
      self.ptr = ffi.gc(lib.ggml_init(params[0]), lib.ggml_free)

  def __op(self, fn, *args, inplace=False) -> GgmlTensor:
      res = fn(self.ptr, *[o.ptr if type(o) == GgmlTensor else o for o in args])
      assert(len(args) > 0)
      first = args[0]
      if inplace:
          assert(isinstance(first, GgmlTensor))
          first.ptr = res
          return first
      
      if isinstance(first, GgmlTensor):
          tp = first.type
          if lib.ggml_is_quantized(tp):
            # tp_size = lib.ggml_type_size(tp)
            nbytes = lib.ggml_nbytes(res)
            nelements = lib.ggml_nelements(res)
            if nbytes == nelements * 4:
                print(f"Warning: nbytes {nbytes} == nelements ({nelements}) * 4 = {nelements * 4}. Dropping to F32.")
                tp = lib.GGML_TYPE_F32


          # if lib.ggml_is_quantized(tp):
              # raise NotImplementedError("Quantized tensors are not supported yet")
          # if lib.ggml_is_quantized(tp) and lib.ggml_nbytes(res) != lib.ggml_nbytes(first.ptr):
      else:
          tp = first
          assert(type(tp) == int)
      
      return GgmlTensor(self, tp, ptr=res)

  def set_param(self, tensor):
    lib.ggml_set_param(self.ptr, tensor.ptr)

  def build_forward(self, tensor):
    return GgmlComputationGraph(self, lib.ggml_build_forward_ctx(self.ptr, tensor.ptr))
  
  def new_graph(self):
    return GgmlComputationGraph(self, lib.ggml_new_graph(self.ptr))

  def new_tensor(self, type, *shape) -> GgmlTensor:
    n_dims = len(shape)
    dims = ffi.new(f'int64_t[]', n_dims)
    for i, dim in enumerate(shape):
      dims[i] = dim
    return self.__op(lib.ggml_new_tensor, type, n_dims, dims)

  def new_tensor_1d(self, type, ne0): return self.__op(lib.ggml_new_tensor_1d, type, ne0)
  def new_tensor_2d(self, type, ne0, ne1): return self.__op(lib.ggml_new_tensor_2d, type, ne0, ne1)
  def new_tensor_3d(self, type, ne0, ne1, ne2): return self.__op(lib.ggml_new_tensor_3d, type, ne0, ne1, ne2)
  def new_tensor_4d(self, type, ne0, ne1, ne2, ne3): return self.__op(lib.ggml_new_tensor_4d, type, ne0, ne1, ne2, ne3)
  
  def tensor(self, shape, dtype=np.float32) -> GgmlTensor:
     type = self.dtype_map.get(dtype) or dtype
     return GgmlTensor(self, type=type, shape=shape)
  
  def dup(self, a): return self.__op(lib.ggml_dup, a)
  def dup_inplace(self, a): return self.__op(lib.ggml_dup_inplace, a, inplace=True)
  def sqr(self, a): return self.__op(lib.ggml_sqr, a)
  def sqr_inplace(self, a): return self.__op(lib.ggml_sqr_inplace, a, inplace=True)
  def sqrt(self, a): return self.__op(lib.ggml_sqrt, a)
  def sqrt_inplace(self, a): return self.__op(lib.ggml_sqrt_inplace, a, inplace=True)
  def log(self, a): return self.__op(lib.ggml_log, a)
  def log_inplace(self, a): return self.__op(lib.ggml_log_inplace, a, inplace=True)
  def sum(self, a): return self.__op(lib.ggml_sum, a)
  def sum_rows(self, a): return self.__op(lib.ggml_sum_rows, a)
  def mean(self, a): return self.__op(lib.ggml_mean, a)
  def argmax(self, a): return self.__op(lib.ggml_argmax, a)
  def abs(self, a): return self.__op(lib.ggml_abs, a)
  def abs_inplace(self, a): return self.__op(lib.ggml_abs_inplace, a, inplace=True)
  def sgn(self, a): return self.__op(lib.ggml_sgn, a)
  def sgn_inplace(self, a): return self.__op(lib.ggml_sgn_inplace, a, inplace=True)
  def neg(self, a): return self.__op(lib.ggml_neg, a)
  def neg_inplace(self, a): return self.__op(lib.ggml_neg_inplace, a, inplace=True)
  def step(self, a): return self.__op(lib.ggml_step, a)
  def step_inplace(self, a): return self.__op(lib.ggml_step_inplace, a, inplace=True)
  def tanh(self, a): return self.__op(lib.ggml_tanh, a)
  def tanh_inplace(self, a): return self.__op(lib.ggml_tanh_inplace, a, inplace=True)
  def elu(self, a): return self.__op(lib.ggml_elu, a)
  def elu_inplace(self, a): return self.__op(lib.ggml_elu_inplace, a, inplace=True)
  def relu(self, a): return self.__op(lib.ggml_relu, a)
  def relu_inplace(self, a): return self.__op(lib.ggml_relu_inplace, a, inplace=True)
  def gelu(self, a): return self.__op(lib.ggml_gelu, a)
  def gelu_inplace(self, a): return self.__op(lib.ggml_gelu_inplace, a, inplace=True)
  def gelu_quick(self, a): return self.__op(lib.ggml_gelu_quick, a)
  def gelu_quick_inplace(self, a): return self.__op(lib.ggml_gelu_quick_inplace, a, inplace=True)
  def silu(self, a): return self.__op(lib.ggml_silu, a)
  def silu_inplace(self, a): return self.__op(lib.ggml_silu_inplace, a, inplace=True)
  def norm(self, a): return self.__op(lib.ggml_norm, a)
  def norm_inplace(self, a): return self.__op(lib.ggml_norm_inplace, a, inplace=True)
  def cont(self, a): return self.__op(lib.ggml_cont, a)
  def cont_inplace(self, a): return self.__op(lib.ggml_cont_inplace, a, inplace=True)
  def transpose(self, a): return self.__op(lib.ggml_transpose, a)
  def soft_max(self, a): return self.__op(lib.ggml_soft_max, a)
  def soft_max_inplace(self, a): return self.__op(lib.ggml_soft_max_inplace, a, inplace=True)

  def add(self, a, b): return self.__op(lib.ggml_add, a, b)
  def add_inplace(self, a, b): return self.__op(lib.ggml_add_inplace, a, b, inplace=True)
  def add1(self, a, b): return self.__op(lib.ggml_add1, a, b)
  def add1_inplace(self, a, b): return self.__op(lib.ggml_add1_inplace, a, b, inplace=True)
  def sub(self, a, b): return self.__op(lib.ggml_sub, a, b)
  def sub_inplace(self, a, b): return self.__op(lib.ggml_sub_inplace, a, b, inplace=True)
  def mul(self, a, b): return self.__op(lib.ggml_mul, a, b)
  def mul_inplace(self, a, b): return self.__op(lib.ggml_mul_inplace, a, b, inplace=True)
  def div(self, a, b): return self.__op(lib.ggml_div, a, b)
  def div_inplace(self, a, b): return self.__op(lib.ggml_div_inplace, a, b, inplace=True)
  def repeat(self, a, b): return self.__op(lib.ggml_repeat, a, b)
  def repeat_back(self, a, b): return self.__op(lib.ggml_repeat_back, a, b)
  def silu_back(self, a, b): return self.__op(lib.ggml_silu_back, a, b)
  def rms_norm_back(self, a, b): return self.__op(lib.ggml_rms_norm_back, a, b)
  def mul_mat(self, a, b): return self.__op(lib.ggml_mul_mat, a, b)
  def out_prod(self, a, b): return self.__op(lib.ggml_out_prod, a, b)
  def scale(self, a, b): return self.__op(lib.ggml_scale, a, b)
  def scale_inplace(self, a, b): return self.__op(lib.ggml_scale_inplace, a, b, inplace=True)
  def cpy(self, a, b): return self.__op(lib.ggml_cpy, a, b)
  def cpy_inplace(self, a, b): return self.__op(lib.ggml_cpy_inplace, a, b, inplace=True)
  def reshape(self, a, b): return self.__op(lib.ggml_reshape, a, b)
  def get_rows(self, a, b): return self.__op(lib.ggml_get_rows, a, b)
  def soft_max_back(self, a, b): return self.__op(lib.ggml_soft_max_back, a, b)
  def soft_max_back_inplace(self, a, b): return self.__op(lib.ggml_soft_max_back_inplace, a, b, inplace=True)

  def rms_norm(self, a, eps): return self.__op(lib.ggml_soft_max_back, a, eps)
  def rms_norm_inplace(self, a, eps): return self.__op(lib.ggml_soft_max_back, a, eps, inplace=True)
    
  def get_rows_back(self, a, b, c): return self.__op(lib.ggml_get_rows_back, a, b, c)
  def diag(self, a): return self.__op(lib.ggml_diag, a)
  def diag_mask_inf(self, a, n_past): return self.__op(lib.ggml_diag_mask_inf, a, n_past)
  def diag_mask_inf_inplace(self, a, n_past): return self.__op(lib.ggml_diag_mask_inf_inplace, a, n_past, inplace=True)
  def diag_mask_zero(self, a, n_past): return self.__op(lib.ggml_diag_mask_zero, a, n_past)
  def diag_mask_zero_inplace(self, a, n_past): return self.__op(lib.ggml_diag_mask_zero_inplace, a, n_past, inplace=True)
  
  def rope(self, a, n_past, n_dims, mode, n_ctx): return self.__op(lib.ggml_rope, a, n_past, n_dims, mode, n_ctx)
  def rope_inplace(self, a, n_past, n_dims, mode, n_ctx): return self.__op(lib.ggml_rope_inplace, a, n_past, n_dims, mode, n_ctx, inplace=True)
  def rope_back(self, a, n_past, n_dims, mode, n_ctx): return self.__op(lib.ggml_rope_back, a, n_past, n_dims, mode, n_ctx)
  
  def rope_custom(self, a, n_past, n_dims, mode, n_ctx, freq_base, freq_scale): return self.__op(lib.ggml_rope_custom, a, n_past, n_dims, mode, n_ctx, freq_base, freq_scale)
  def rope_custom_inplace(self, a, n_past, n_dims, mode, n_ctx, freq_base, freq_scale): return self.__op(lib.ggml_rope_custom_inplace, a, n_past, n_dims, mode, n_ctx, freq_base, freq_scale, inplace=True)
  
  def alibi(self, a, n_past, n_head, bias_max): return self.__op(lib.ggml_alibi, a, n_past, n_head, bias_max)
  def clamp(self, a, min, max): return self.__op(lib.ggml_clamp, a, min, max)

  def conv_1d(self, a, b, s0, p0, d0): return self.__op(lib.ggml_conv_1d, a, b, s0, p0, d0)
  def conv_1d_ph(self, a, b, s, d): return self.__op(lib.ggml_conv_1d_ph, a, b, s, d)
  def conv_2d(self, a, b, s0, s1, p0, p1, d0, d1): return self.__op(lib.ggml_conv_1d, a, b, s0, s1, p0, p1, d0, d1)
  
  def pool_1d(self, a, op, k0, s0, p0): return self.__op(lib.ggml_pool_1d, a, op, k0, s0, p0)
  def pool_2d(self, a, op, k0, k1, s0, s1, p0, p1): return self.__op(lib.ggml_pool_2d, a, op, k0, k1, s0, s1, p0, p1)

  def set(self, a, b, nb1, nb2, nb3, offset): return self.__op(lib.ggml_set, a, b, nb1, nb2, nb3, offset)
  def set_inplace(self, a, b, nb1, nb2, nb3, offset): return self.__op(lib.ggml_set_inplace, a, b, nb1, nb2, nb3, offset, inplace=TrueTrue)
  def set_1d(self, a, b, offset): return self.__op(lib.ggml_set_1d, a, b, offset)
  def set_1d_inplace(self, a, b, offset): return self.__op(lib.ggml_set_1d_inplace, a, b, offset, inplace=True)

  def set_2d(self, a, b, nb1, offset): return self.__op(lib.ggml_set_2d, a, b, nb1, offset)

  def set_2d_inplace(self, a, b, nb1, offset): return self.__op(lib.ggml_set_2d_inplace, a, b, nb1, offset, inplace=True)

  def view_1d(self, a, ne0, offset): return self.__op(lib.ggml_view_1d, a, ne0, offset)
  def view_2d(self, a, ne0, ne1, nb1, offset): return self.__op(lib.ggml_view_2d, a, ne0, ne1, nb1, offset)
  def view_3d(self, a, ne0, ne1, ne2, nb1, nb2, offset): return self.__op(lib.ggml_view_3d, a, ne0, ne1, ne2, nb1, nb2, offset)
  def view_4d(self, a, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset): return self.__op(lib.ggml_view_4d, a, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset)

  def win_part(self, a, w): return self.__op(lib.ggml_win_part, a, w)
  def win_unpart(self, a, w0, h0, w): return self.__op(lib.ggml_win_unpart, a, w0, h0, w)

  def unary(self, a, op): return self.__op(lib.ggml_unary, a, op)
  def unary_inplace(self, a, op): return self.__op(lib.ggml_unary_inplace, a, op, inplace=True)
  def map_unary_f32(self, a, fun): return self.__op(lib.ggml_map_unary_f32, a, fun)
  def map_unary_inplace_f32(self, a, fun): return self.__op(lib.ggml_map_unary_inplace_f32, a, fun)
  def map_binary_f32(self, a, b, fun): return self.__op(lib.ggml_map_binary_f32, a, b, fun)
  def map_binary_inplace_f32(self, a, b, fun): return self.__op(lib.ggml_map_binary_inplace_f32, a, b, fun)
  def map_custom1_f32(self, a, fun): return self.__op(lib.ggml_map_custom1_f32, a, fun)
  def map_custom1_inplace_f32(self, a, fun): return self.__op(lib.ggml_map_custom1_inplace_f32, a, fun)
  def map_custom2_f32(self, a, b, fun): return self.__op(lib.ggml_map_custom2_f32, a, b, fun)
  def map_custom2_inplace_f32(self, a, b, fun): return self.__op(lib.ggml_map_custom2_inplace_f32, a, b, fun)
  def map_custom3_f32(self, a, b, c, fun): return self.__op(lib.ggml_map_custom3_f32, a, b, c, fun)
  def map_custom3_inplace_f32(self, a, b, c, fun): return self.__op(lib.ggml_map_custom3_inplace_f32, a, b, c, fun)
    
  def cross_entropy_loss(self, a, b): return self.__op(lib.ggml_cross_entropy_loss, a, b)
  def cross_entropy_loss_back(self, a, b, c): return self.__op(lib.ggml_cross_entropy_loss_back, a, b, c)
    
class GgmlComputationGraph:
  def __init__(self, ctx, ptr):
    self.ctx = ctx
    self.ptr = ptr

  def build_forward_expand(self, t) -> None:
    lib.ggml_build_forward_expand(self.ptr, t.ptr)

  def build_backward(self, keep: bool):
    return GgmlComputationGraph(self.ctx, lib.ggml_build_backward(self.ctx.ptr, self.ptr, keep))


  def compute(self, n_threads) -> None:
    lib.ggml_graph_compute_with_ctx(self.ctx.ptr, self.ptr, n_threads)
