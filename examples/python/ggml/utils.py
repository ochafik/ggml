"""
  Common helpers for working with ggml tensors and numpy arrays.
"""
from ggml import ffi, lib
from typing import Union
import numpy as np

TensorLike = Union[ffi.CData, np.ndarray]

def init(mem_size: int, mem_buffer: ffi.CData = ffi.NULL, no_alloc: bool = False) -> ffi.CData:
    """
      Initialize a ggml context, which will be freed automatically when the pointer is garbage collected.
    """
    params = ffi.new('struct ggml_init_params*')
    params.mem_size = mem_size
    params.mem_buffer = mem_buffer
    params.no_alloc = no_alloc
    return ffi.gc(lib.ggml_init(params[0]), lib.ggml_free)
    
def copy(from_tensor: TensorLike, to_tensor: TensorLike, allow_requantize: bool = True):
    """
      Copy the contents of one tensor to another, doing any necessary type / quantization conversions.
      Works across numpy & ggml tensors, but they must have the same shape (and be contiguous).

      Parameters
      ----------
      from_tensor : TensorLike
          The tensor to copy from (a numpy array or ggml tensor, quantized or not)
      to_tensor : TensorLike
          The tensor to copy to (a numpy array or ggml tensor, quantized or not)
      allow_requantize : bool
          If False, will throw an error if requantization is required (i.e. both from_tensor
          and to_tensor are quantized with different quantization types)
    """
    if id(from_tensor) == id(to_tensor):
        return
    
    __expect_same_layout("source", from_tensor, "destination", to_tensor)

    __check_shape_consistent_with_type(from_tensor)
    __check_shape_consistent_with_type(to_tensor)

    from_type = __get_type(from_tensor)
    to_type = __get_type(to_tensor)

    if from_type == to_type:
        ffi.memmove(__get_data(to_tensor), __get_data(from_tensor), __get_nbytes(from_tensor))
    else:
        if lib.ggml_is_quantized(from_type) and lib.ggml_is_quantized(to_type) and not allow_requantize:
            raise ValueError(f"Requantizing from type {from_type} to {to_type} is disabled. Force with allow_requantize=True")    
        
        f32_data = __get_floats(from_tensor)
        __set_floats(to_tensor, f32_data)

def numpy(tensor: ffi.CData, allow_copy: Union[bool, np.ndarray] = False, allow_requantize=False) -> np.ndarray:
    """
      Convert a ggml tensor to a numpy array.
      If the tensor isn't quantized, the returned numpy array will be a view over its data.
      
      If it is quantized (and allow_copy is True), the copy will involve dequantization and the returned array will
      be a copy of the original tensor (any changes to the numpy array won't then be reflected back to the tensor).

      Parameters
      ----------
      tensor : ffi.CData
          The tensor to convert to a numpy array
      allow_copy : bool or np.ndarray
          If False, will throw an error if the tensor is quantized (since dequantization requires extra memory).
          If True, will dequantize the tensor and return a copy of the data in a new float32 numpy array.
          If an np.ndarray, will copy the data into the given array (which must be the same shape as the tensor) when dequantization is needed
      allow_requantize : bool
          If allow_copy is a tensor with a different quantization type than the source tensor, will throw an error unless allow_requantize is True.
    """
    shape = __get_shape(tensor)

    if isinstance(allow_copy, np.ndarray):
        __expect_same_layout("source tensor", tensor, "dequantization output tensor", allow_copy)

    data = lib.ggml_get_data(tensor)

    if lib.ggml_is_quantized(tensor.type):
        if allow_copy == False:
            raise ValueError(f"{__describe(tensor)} is quantized, conversion to numpy requires a copy (pass allow_copy=True; changes to the numpy array won't affect the original).")
      
        destination = allow_copy if isinstance(allow_copy, np.ndarray) else np.empty(shape, dtype=np.float32)
        copy(tensor, destination, allow_requantize=allow_requantize)
        return destination
    else:
        dtype = type_to_dtype(tensor.type)
        if not dtype:
            raise NotImplementedError(f'Cannot convert {__describe(tensor)} to numpy')

        strides = __get_strides(tensor)
        actualsize = strides[-1] * shape[-1]
        a = np.frombuffer(ffi.buffer(data, actualsize), dtype=dtype)
        a.shape = shape
        return a

def __type_name(type: int) -> str:
    name = lib.ggml_type_name(type)
    return ffi.string(name).decode('utf-8') if name else None

__k_quant_types = set([
  lib.GGML_TYPE_Q2_K,
  lib.GGML_TYPE_Q3_K,
  lib.GGML_TYPE_Q4_K,
  lib.GGML_TYPE_Q5_K,
  lib.GGML_TYPE_Q6_K,
  lib.GGML_TYPE_Q8_K,
])

__type_to_dtype_dict = {
  lib.GGML_TYPE_I8: np.int8,
  lib.GGML_TYPE_I16: np.int16,
  lib.GGML_TYPE_I32: np.int32,
  lib.GGML_TYPE_F16: np.float16,
  lib.GGML_TYPE_F32: np.float32,
}

# TODO: remove
def type_to_dtype(type: int):
  return __type_to_dtype_dict.get(type)

def __describe(tensor: ffi.CType) -> str:
    shape = __get_shape(tensor)
    t = __type_name(tensor.type)
    return f'Tensor[{t}, {shape}]' 
    # return lib.ggml_type_name(type)

# Note: dtype doesn't seem hashable
def __dtype_to_type(dtype: np.dtype):
    if dtype == np.float32: return lib.GGML_TYPE_F32
    elif dtype == np.float16: return lib.GGML_TYPE_F16
    elif dtype == np.int32: return lib.GGML_TYPE_I32
    elif dtype == np.int16: return lib.GGML_TYPE_I16
    elif dtype == np.int8: return lib.GGML_TYPE_I8
    else: raise ValueError(f"Unsupported dtype: {dtype}")

def __get_type(tensor: TensorLike): return __dtype_to_type(tensor.dtype) if isinstance(tensor, np.ndarray) else tensor.type
def __get_shape(x: TensorLike): return x.shape if isinstance(x, np.ndarray) else tuple([x.ne[i] for i in range(x.n_dims)])
def __get_strides(x: TensorLike): return x.strides if isinstance(x, np.ndarray) else tuple([x.nb[i] for i in range(x.n_dims)])
def __get_data(x: TensorLike) -> ffi.CData: return ffi.from_buffer(x) if isinstance(x, np.ndarray) else lib.ggml_get_data(x)
def __get_nbytes(tensor: TensorLike): return tensor.nbytes if isinstance(tensor, np.ndarray) else lib.ggml_nbytes(tensor)
def __get_nelements(tensor: TensorLike): return tensor.size if isinstance(tensor, np.ndarray) else lib.ggml_nelements(tensor)

# TODO(ochafik): Check strides too / is each tensor contiguous?
def __expect_same_layout(name1: str, tensor1: TensorLike, name2: str, tensor2:   TensorLike):
    shape1, shape2 = __get_shape(tensor1), __get_shape(tensor2)
    if shape1 != shape2:
        raise ValueError(f"Shape mismatch: {name1} has {shape1} but {name2} has {shape2}")
    
def __get_floats(tensor: TensorLike) -> ffi.CData:
    data, type = __get_data(tensor), __get_type(tensor)
    if type == lib.GGML_TYPE_F32:
        return ffi.cast('float*', data)
    else:
      nelements = __get_nelements(tensor)
      floats = ffi.new('float[]', nelements)
      if type == lib.GGML_TYPE_F16:
          lib.ggml_fp16_to_fp32_row(data, floats, nelements)
      elif lib.ggml_is_quantized(type):
          qtype = lib.ggml_internal_get_type_traits(type)
          assert qtype.to_float, f"Type {__type_name(type)} is not supported by ggml"
          qtype.to_float(data, floats, nelements)
      else:
          raise NotImplementedError(f'Cannot read floats from {__describe(tensor)}')
      return floats

def __set_floats(tensor: TensorLike, f32_data: ffi.CData) -> None:
    data, type, nbytes = __get_data(tensor), __get_type(tensor), __get_nbytes(tensor)
    if type == lib.GGML_TYPE_F32:
        ffi.memmove(data, f32_data, nbytes)
    else:
      nelements = __get_nelements(tensor)
      if type == lib.GGML_TYPE_F16:
          lib.ggml_fp32_to_fp16_row(f32_data, data, nelements)
      elif lib.ggml_is_quantized(type):
          qtype = lib.ggml_internal_get_type_traits(type)
          assert qtype.from_float, f"Type {__type_name(type)} is not supported by ggml"
          qtype.from_float(f32_data, data, nelements)
      else:
          raise NotImplementedError(f'Cannot write floats to {__describe(tensor)}')

def __check_shape_consistent_with_type(tensor: ffi.CData):
    type = __get_type(tensor)
    if not lib.ggml_is_quantized(type):
        return
    shape = __get_shape(tensor)
    if len(shape) > 2:
        raise ValueError(f"No support for quantized tensors with {tensor.n_dims} dimensions")
    block_size = lib.ggml_blck_size(type)
    if block_size == 0 and type in __k_quant_types:
        raise ValueError(f"Can't quantize, native library was not compiled with USE_K_QUANTS!")
    assert(block_size > 0)
    if (shape[0] % block_size != 0 or (len(shape) > 1 and shape[1] % block_size != 0)):
        raise ValueError(f"Tensor sizes {shape} are not divisible by {block_size}, required for quantization.")

