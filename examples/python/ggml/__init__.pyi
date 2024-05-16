# auto-generated file
import ggml.ffi as ffi
import numpy as np
class lib:
  @property
  def GGML_BACKEND_BUFFER_USAGE_ANY(self) -> int: ...
  @property
  def GGML_BACKEND_BUFFER_USAGE_WEIGHTS(self) -> int: ...
  @property
  def GGML_BACKEND_TYPE_CPU(self) -> int: ...
  @property
  def GGML_BACKEND_TYPE_GPU(self) -> int: ...
  @property
  def GGML_BACKEND_TYPE_GPU_SPLIT(self) -> int: ...
  @property
  def GGML_CGRAPH_EVAL_ORDER_COUNT(self) -> int: ...
  @property
  def GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT(self) -> int: ...
  @property
  def GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT(self) -> int: ...
  @property
  def GGML_FTYPE_ALL_F32(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_BF16(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_F16(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_IQ1_M(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_IQ1_S(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_IQ2_S(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_IQ2_XS(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_IQ2_XXS(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_IQ3_S(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_IQ3_XXS(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_IQ4_NL(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_IQ4_XS(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q2_K(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q3_K(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q4_0(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q4_1(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q4_1_SOME_F16(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q4_K(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q5_0(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q5_1(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q5_K(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q6_K(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q8_0(self) -> int: ...
  @property
  def GGML_FTYPE_UNKNOWN(self) -> int: ...
  @property
  def GGML_LINESEARCH_BACKTRACKING_ARMIJO(self) -> int: ...
  @property
  def GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE(self) -> int: ...
  @property
  def GGML_LINESEARCH_BACKTRACKING_WOLFE(self) -> int: ...
  @property
  def GGML_LINESEARCH_DEFAULT(self) -> int: ...
  @property
  def GGML_LINESEARCH_FAIL(self) -> int: ...
  @property
  def GGML_LINESEARCH_INVALID_PARAMETERS(self) -> int: ...
  @property
  def GGML_LINESEARCH_MAXIMUM_ITERATIONS(self) -> int: ...
  @property
  def GGML_LINESEARCH_MAXIMUM_STEP(self) -> int: ...
  @property
  def GGML_LINESEARCH_MINIMUM_STEP(self) -> int: ...
  @property
  def GGML_LOG_LEVEL_DEBUG(self) -> int: ...
  @property
  def GGML_LOG_LEVEL_ERROR(self) -> int: ...
  @property
  def GGML_LOG_LEVEL_INFO(self) -> int: ...
  @property
  def GGML_LOG_LEVEL_WARN(self) -> int: ...
  @property
  def GGML_NUMA_STRATEGY_COUNT(self) -> int: ...
  @property
  def GGML_NUMA_STRATEGY_DISABLED(self) -> int: ...
  @property
  def GGML_NUMA_STRATEGY_DISTRIBUTE(self) -> int: ...
  @property
  def GGML_NUMA_STRATEGY_ISOLATE(self) -> int: ...
  @property
  def GGML_NUMA_STRATEGY_MIRROR(self) -> int: ...
  @property
  def GGML_NUMA_STRATEGY_NUMACTL(self) -> int: ...
  @property
  def GGML_OBJECT_TYPE_GRAPH(self) -> int: ...
  @property
  def GGML_OBJECT_TYPE_TENSOR(self) -> int: ...
  @property
  def GGML_OBJECT_TYPE_WORK_BUFFER(self) -> int: ...
  @property
  def GGML_OPT_RESULT_CANCEL(self) -> int: ...
  @property
  def GGML_OPT_RESULT_DID_NOT_CONVERGE(self) -> int: ...
  @property
  def GGML_OPT_RESULT_FAIL(self) -> int: ...
  @property
  def GGML_OPT_RESULT_INVALID_WOLFE(self) -> int: ...
  @property
  def GGML_OPT_RESULT_NO_CONTEXT(self) -> int: ...
  @property
  def GGML_OPT_RESULT_OK(self) -> int: ...
  @property
  def GGML_OPT_TYPE_ADAM(self) -> int: ...
  @property
  def GGML_OPT_TYPE_LBFGS(self) -> int: ...
  @property
  def GGML_OP_ACC(self) -> int: ...
  @property
  def GGML_OP_ADD(self) -> int: ...
  @property
  def GGML_OP_ADD1(self) -> int: ...
  @property
  def GGML_OP_ADD_REL_POS(self) -> int: ...
  @property
  def GGML_OP_ARANGE(self) -> int: ...
  @property
  def GGML_OP_ARGMAX(self) -> int: ...
  @property
  def GGML_OP_ARGSORT(self) -> int: ...
  @property
  def GGML_OP_CLAMP(self) -> int: ...
  @property
  def GGML_OP_CONCAT(self) -> int: ...
  @property
  def GGML_OP_CONT(self) -> int: ...
  @property
  def GGML_OP_CONV_TRANSPOSE_1D(self) -> int: ...
  @property
  def GGML_OP_CONV_TRANSPOSE_2D(self) -> int: ...
  @property
  def GGML_OP_COUNT(self) -> int: ...
  @property
  def GGML_OP_CPY(self) -> int: ...
  @property
  def GGML_OP_CROSS_ENTROPY_LOSS(self) -> int: ...
  @property
  def GGML_OP_CROSS_ENTROPY_LOSS_BACK(self) -> int: ...
  @property
  def GGML_OP_DIAG(self) -> int: ...
  @property
  def GGML_OP_DIAG_MASK_INF(self) -> int: ...
  @property
  def GGML_OP_DIAG_MASK_ZERO(self) -> int: ...
  @property
  def GGML_OP_DIV(self) -> int: ...
  @property
  def GGML_OP_DUP(self) -> int: ...
  @property
  def GGML_OP_FLASH_ATTN(self) -> int: ...
  @property
  def GGML_OP_FLASH_ATTN_BACK(self) -> int: ...
  @property
  def GGML_OP_FLASH_ATTN_EXT(self) -> int: ...
  @property
  def GGML_OP_FLASH_FF(self) -> int: ...
  @property
  def GGML_OP_GET_REL_POS(self) -> int: ...
  @property
  def GGML_OP_GET_ROWS(self) -> int: ...
  @property
  def GGML_OP_GET_ROWS_BACK(self) -> int: ...
  @property
  def GGML_OP_GROUP_NORM(self) -> int: ...
  @property
  def GGML_OP_IM2COL(self) -> int: ...
  @property
  def GGML_OP_LEAKY_RELU(self) -> int: ...
  @property
  def GGML_OP_LOG(self) -> int: ...
  @property
  def GGML_OP_MAP_BINARY(self) -> int: ...
  @property
  def GGML_OP_MAP_CUSTOM1(self) -> int: ...
  @property
  def GGML_OP_MAP_CUSTOM1_F32(self) -> int: ...
  @property
  def GGML_OP_MAP_CUSTOM2(self) -> int: ...
  @property
  def GGML_OP_MAP_CUSTOM2_F32(self) -> int: ...
  @property
  def GGML_OP_MAP_CUSTOM3(self) -> int: ...
  @property
  def GGML_OP_MAP_CUSTOM3_F32(self) -> int: ...
  @property
  def GGML_OP_MAP_UNARY(self) -> int: ...
  @property
  def GGML_OP_MEAN(self) -> int: ...
  @property
  def GGML_OP_MUL(self) -> int: ...
  @property
  def GGML_OP_MUL_MAT(self) -> int: ...
  @property
  def GGML_OP_MUL_MAT_ID(self) -> int: ...
  @property
  def GGML_OP_NONE(self) -> int: ...
  @property
  def GGML_OP_NORM(self) -> int: ...
  @property
  def GGML_OP_OUT_PROD(self) -> int: ...
  @property
  def GGML_OP_PAD(self) -> int: ...
  @property
  def GGML_OP_PERMUTE(self) -> int: ...
  @property
  def GGML_OP_POOL_1D(self) -> int: ...
  @property
  def GGML_OP_POOL_2D(self) -> int: ...
  @property
  def GGML_OP_POOL_AVG(self) -> int: ...
  @property
  def GGML_OP_POOL_COUNT(self) -> int: ...
  @property
  def GGML_OP_POOL_MAX(self) -> int: ...
  @property
  def GGML_OP_REPEAT(self) -> int: ...
  @property
  def GGML_OP_REPEAT_BACK(self) -> int: ...
  @property
  def GGML_OP_RESHAPE(self) -> int: ...
  @property
  def GGML_OP_RMS_NORM(self) -> int: ...
  @property
  def GGML_OP_RMS_NORM_BACK(self) -> int: ...
  @property
  def GGML_OP_ROPE(self) -> int: ...
  @property
  def GGML_OP_ROPE_BACK(self) -> int: ...
  @property
  def GGML_OP_SCALE(self) -> int: ...
  @property
  def GGML_OP_SET(self) -> int: ...
  @property
  def GGML_OP_SILU_BACK(self) -> int: ...
  @property
  def GGML_OP_SOFT_MAX(self) -> int: ...
  @property
  def GGML_OP_SOFT_MAX_BACK(self) -> int: ...
  @property
  def GGML_OP_SQR(self) -> int: ...
  @property
  def GGML_OP_SQRT(self) -> int: ...
  @property
  def GGML_OP_SSM_CONV(self) -> int: ...
  @property
  def GGML_OP_SSM_SCAN(self) -> int: ...
  @property
  def GGML_OP_SUB(self) -> int: ...
  @property
  def GGML_OP_SUM(self) -> int: ...
  @property
  def GGML_OP_SUM_ROWS(self) -> int: ...
  @property
  def GGML_OP_TIMESTEP_EMBEDDING(self) -> int: ...
  @property
  def GGML_OP_TRANSPOSE(self) -> int: ...
  @property
  def GGML_OP_UNARY(self) -> int: ...
  @property
  def GGML_OP_UPSCALE(self) -> int: ...
  @property
  def GGML_OP_VIEW(self) -> int: ...
  @property
  def GGML_OP_WIN_PART(self) -> int: ...
  @property
  def GGML_OP_WIN_UNPART(self) -> int: ...
  @property
  def GGML_PREC_DEFAULT(self) -> int: ...
  @property
  def GGML_PREC_F32(self) -> int: ...
  @property
  def GGML_SORT_ORDER_ASC(self) -> int: ...
  @property
  def GGML_SORT_ORDER_DESC(self) -> int: ...
  @property
  def GGML_STATUS_ABORTED(self) -> int: ...
  @property
  def GGML_STATUS_ALLOC_FAILED(self) -> int: ...
  @property
  def GGML_STATUS_FAILED(self) -> int: ...
  @property
  def GGML_STATUS_SUCCESS(self) -> int: ...
  @property
  def GGML_TASK_TYPE_COMPUTE(self) -> int: ...
  @property
  def GGML_TASK_TYPE_FINALIZE(self) -> int: ...
  @property
  def GGML_TASK_TYPE_INIT(self) -> int: ...
  @property
  def GGML_TENSOR_FLAG_INPUT(self) -> int: ...
  @property
  def GGML_TENSOR_FLAG_OUTPUT(self) -> int: ...
  @property
  def GGML_TENSOR_FLAG_PARAM(self) -> int: ...
  @property
  def GGML_TYPE_BF16(self) -> int: ...
  @property
  def GGML_TYPE_COUNT(self) -> int: ...
  @property
  def GGML_TYPE_F16(self) -> int: ...
  @property
  def GGML_TYPE_F32(self) -> int: ...
  @property
  def GGML_TYPE_F64(self) -> int: ...
  @property
  def GGML_TYPE_I16(self) -> int: ...
  @property
  def GGML_TYPE_I32(self) -> int: ...
  @property
  def GGML_TYPE_I64(self) -> int: ...
  @property
  def GGML_TYPE_I8(self) -> int: ...
  @property
  def GGML_TYPE_IQ1_M(self) -> int: ...
  @property
  def GGML_TYPE_IQ1_S(self) -> int: ...
  @property
  def GGML_TYPE_IQ2_S(self) -> int: ...
  @property
  def GGML_TYPE_IQ2_XS(self) -> int: ...
  @property
  def GGML_TYPE_IQ2_XXS(self) -> int: ...
  @property
  def GGML_TYPE_IQ3_S(self) -> int: ...
  @property
  def GGML_TYPE_IQ3_XXS(self) -> int: ...
  @property
  def GGML_TYPE_IQ4_NL(self) -> int: ...
  @property
  def GGML_TYPE_IQ4_XS(self) -> int: ...
  @property
  def GGML_TYPE_Q2_K(self) -> int: ...
  @property
  def GGML_TYPE_Q3_K(self) -> int: ...
  @property
  def GGML_TYPE_Q4_0(self) -> int: ...
  @property
  def GGML_TYPE_Q4_1(self) -> int: ...
  @property
  def GGML_TYPE_Q4_K(self) -> int: ...
  @property
  def GGML_TYPE_Q5_0(self) -> int: ...
  @property
  def GGML_TYPE_Q5_1(self) -> int: ...
  @property
  def GGML_TYPE_Q5_K(self) -> int: ...
  @property
  def GGML_TYPE_Q6_K(self) -> int: ...
  @property
  def GGML_TYPE_Q8_0(self) -> int: ...
  @property
  def GGML_TYPE_Q8_1(self) -> int: ...
  @property
  def GGML_TYPE_Q8_K(self) -> int: ...
  @property
  def GGML_UNARY_OP_ABS(self) -> int: ...
  @property
  def GGML_UNARY_OP_COUNT(self) -> int: ...
  @property
  def GGML_UNARY_OP_ELU(self) -> int: ...
  @property
  def GGML_UNARY_OP_GELU(self) -> int: ...
  @property
  def GGML_UNARY_OP_GELU_QUICK(self) -> int: ...
  @property
  def GGML_UNARY_OP_HARDSIGMOID(self) -> int: ...
  @property
  def GGML_UNARY_OP_HARDSWISH(self) -> int: ...
  @property
  def GGML_UNARY_OP_NEG(self) -> int: ...
  @property
  def GGML_UNARY_OP_RELU(self) -> int: ...
  @property
  def GGML_UNARY_OP_SGN(self) -> int: ...
  @property
  def GGML_UNARY_OP_SIGMOID(self) -> int: ...
  @property
  def GGML_UNARY_OP_SILU(self) -> int: ...
  @property
  def GGML_UNARY_OP_STEP(self) -> int: ...
  @property
  def GGML_UNARY_OP_TANH(self) -> int: ...
  @property
  def GGUF_TYPE_ARRAY(self) -> int: ...
  @property
  def GGUF_TYPE_BOOL(self) -> int: ...
  @property
  def GGUF_TYPE_COUNT(self) -> int: ...
  @property
  def GGUF_TYPE_FLOAT32(self) -> int: ...
  @property
  def GGUF_TYPE_FLOAT64(self) -> int: ...
  @property
  def GGUF_TYPE_INT16(self) -> int: ...
  @property
  def GGUF_TYPE_INT32(self) -> int: ...
  @property
  def GGUF_TYPE_INT64(self) -> int: ...
  @property
  def GGUF_TYPE_INT8(self) -> int: ...
  @property
  def GGUF_TYPE_STRING(self) -> int: ...
  @property
  def GGUF_TYPE_UINT16(self) -> int: ...
  @property
  def GGUF_TYPE_UINT32(self) -> int: ...
  @property
  def GGUF_TYPE_UINT64(self) -> int: ...
  @property
  def GGUF_TYPE_UINT8(self) -> int: ...
  def dequantize_row_iq1_m(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_iq1_m  (const block_iq1_m   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);"""
    ...
  def dequantize_row_iq1_s(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_iq1_s  (const block_iq1_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);"""
    ...
  def dequantize_row_iq2_s(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_iq2_s  (const block_iq2_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);"""
    ...
  def dequantize_row_iq2_xs(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_iq2_xs (const block_iq2_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);"""
    ...
  def dequantize_row_iq2_xxs(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_iq2_xxs(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);"""
    ...
  def dequantize_row_iq3_s(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_iq3_s  (const block_iq3_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);"""
    ...
  def dequantize_row_iq3_xxs(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_iq3_xxs(const block_iq3_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);"""
    ...
  def dequantize_row_iq4_nl(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_iq4_nl (const block_iq4_nl  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);"""
    ...
  def dequantize_row_iq4_xs(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_iq4_xs (const block_iq4_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);"""
    ...
  def dequantize_row_q2_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q2_K(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);"""
    ...
  def dequantize_row_q3_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q3_K(const block_q3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);"""
    ...
  def dequantize_row_q4_0(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """
    Dequantization

    void dequantize_row_q4_0(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
    """
    ...
  def dequantize_row_q4_1(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q4_1(const block_q4_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);"""
    ...
  def dequantize_row_q4_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q4_K(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);"""
    ...
  def dequantize_row_q5_0(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q5_0(const block_q5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);"""
    ...
  def dequantize_row_q5_1(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q5_1(const block_q5_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);"""
    ...
  def dequantize_row_q5_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q5_K(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);"""
    ...
  def dequantize_row_q6_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);"""
    ...
  def dequantize_row_q8_0(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q8_0(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);"""
    ...
  def dequantize_row_q8_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q8_K(const block_q8_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);"""
    ...
  def ggml_abs(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_abs(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_abs_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_abs_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_acc(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, nb1: int, nb2: int, nb3: int, offset: int) -> ffi.CData:
    """
    dst = a
    view(dst, nb1, nb2, nb3, offset) += b
    return dst

        GGML_API struct ggml_tensor * ggml_acc(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                size_t                nb1,
                size_t                nb2,
                size_t                nb3,
                size_t                offset);
    """
    ...
  def ggml_acc_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, nb1: int, nb2: int, nb3: int, offset: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_acc_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                size_t                nb1,
                size_t                nb2,
                size_t                nb3,
                size_t                offset);
    """
    ...
  def ggml_add(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_add(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_add1(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_add1(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_add1_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_add1_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_add_cast(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, type: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_add_cast(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                enum   ggml_type      type);
    """
    ...
  def ggml_add_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_add_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_add_rel_pos(ctx: ffi.CData, a: ffi.CData, pw: ffi.CData, ph: ffi.CData) -> ffi.CData:
    """
    used in sam

        GGML_API struct ggml_tensor * ggml_add_rel_pos(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * pw,
                struct ggml_tensor  * ph);
    """
    ...
  def ggml_add_rel_pos_inplace(ctx: ffi.CData, a: ffi.CData, pw: ffi.CData, ph: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_add_rel_pos_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * pw,
                struct ggml_tensor  * ph);
    """
    ...
  def ggml_arange(ctx: ffi.CData, start: float, stop: float, step: float) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_arange(
                struct ggml_context * ctx,
                float                 start,
                float                 stop,
                float                 step);
    """
    ...
  def ggml_are_same_shape(t0: ffi.CData, t1: ffi.CData) -> ffi.CData:
    """    GGML_API bool ggml_are_same_shape (const struct ggml_tensor * t0, const struct ggml_tensor * t1);"""
    ...
  def ggml_are_same_stride(t0: ffi.CData, t1: ffi.CData) -> ffi.CData:
    """    GGML_API bool ggml_are_same_stride(const struct ggml_tensor * t0, const struct ggml_tensor * t1);"""
    ...
  def ggml_argmax(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
    argmax along rows

        GGML_API struct ggml_tensor * ggml_argmax(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_argsort(ctx: ffi.CData, a: ffi.CData, order: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_argsort(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                enum ggml_sort_order  order);
    """
    ...
  def ggml_backend_alloc_buffer(backend: ffi.CData, size: int) -> ffi.CData:
    """    GGML_API ggml_backend_buffer_t      ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size);"""
    ...
  def ggml_backend_alloc_ctx_tensors(ctx: ffi.CData, backend: ffi.CData) -> ffi.CData:
    """GGML_API struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors(struct ggml_context * ctx, ggml_backend_t backend);"""
    ...
  def ggml_backend_alloc_ctx_tensors_from_buft(ctx: ffi.CData, buft: ffi.CData) -> ffi.CData:
    """
    Utils
    Create a buffer and allocate all the tensors in a ggml_context

    GGML_API struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft);
    """
    ...
  def ggml_backend_buffer_clear(buffer: ffi.CData, value: int) -> None:
    """    GGML_API           void                       ggml_backend_buffer_clear         (ggml_backend_buffer_t buffer, uint8_t value);"""
    ...
  def ggml_backend_buffer_free(buffer: ffi.CData) -> None:
    """    GGML_API           void                       ggml_backend_buffer_free          (ggml_backend_buffer_t buffer);"""
    ...
  def ggml_backend_buffer_get_alignment(buffer: ffi.CData) -> int:
    """    GGML_API           size_t                     ggml_backend_buffer_get_alignment (ggml_backend_buffer_t buffer);"""
    ...
  def ggml_backend_buffer_get_alloc_size(buffer: ffi.CData, tensor: ffi.CData) -> int:
    """    GGML_API           size_t                     ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);"""
    ...
  def ggml_backend_buffer_get_base(buffer: ffi.CData) -> ffi.CData:
    """    GGML_API           void *                     ggml_backend_buffer_get_base      (ggml_backend_buffer_t buffer);"""
    ...
  def ggml_backend_buffer_get_max_size(buffer: ffi.CData) -> int:
    """    GGML_API           size_t                     ggml_backend_buffer_get_max_size  (ggml_backend_buffer_t buffer);"""
    ...
  def ggml_backend_buffer_get_size(buffer: ffi.CData) -> int:
    """    GGML_API           size_t                     ggml_backend_buffer_get_size      (ggml_backend_buffer_t buffer);"""
    ...
  def ggml_backend_buffer_get_type(buffer: ffi.CData) -> ffi.CData:
    """    GGML_API           ggml_backend_buffer_type_t ggml_backend_buffer_get_type      (ggml_backend_buffer_t buffer);"""
    ...
  def ggml_backend_buffer_init_tensor(buffer: ffi.CData, tensor: ffi.CData) -> None:
    """    GGML_API GGML_CALL void                       ggml_backend_buffer_init_tensor   (ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);"""
    ...
  def ggml_backend_buffer_is_host(buffer: ffi.CData) -> ffi.CData:
    """    GGML_API           bool                       ggml_backend_buffer_is_host       (ggml_backend_buffer_t buffer);"""
    ...
  def ggml_backend_buffer_name(buffer: ffi.CData) -> ffi.CData:
    """    GGML_API           const char *               ggml_backend_buffer_name          (ggml_backend_buffer_t buffer);"""
    ...
  def ggml_backend_buffer_reset(buffer: ffi.CData) -> None:
    """    GGML_API           void                       ggml_backend_buffer_reset         (ggml_backend_buffer_t buffer);"""
    ...
  def ggml_backend_buffer_set_usage(buffer: ffi.CData, usage: int) -> None:
    """    GGML_API           void                       ggml_backend_buffer_set_usage     (ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage);"""
    ...
  def ggml_backend_buft_alloc_buffer(buft: ffi.CData, size: int) -> ffi.CData:
    """    GGML_API GGML_CALL ggml_backend_buffer_t ggml_backend_buft_alloc_buffer    (ggml_backend_buffer_type_t buft, size_t size);"""
    ...
  def ggml_backend_buft_get_alignment(buft: ffi.CData) -> int:
    """    GGML_API           size_t                ggml_backend_buft_get_alignment   (ggml_backend_buffer_type_t buft);"""
    ...
  def ggml_backend_buft_get_alloc_size(buft: ffi.CData, tensor: ffi.CData) -> int:
    """    GGML_API GGML_CALL size_t                ggml_backend_buft_get_alloc_size  (ggml_backend_buffer_type_t buft, struct ggml_tensor * tensor);"""
    ...
  def ggml_backend_buft_get_max_size(buft: ffi.CData) -> int:
    """    GGML_API           size_t                ggml_backend_buft_get_max_size    (ggml_backend_buffer_type_t buft);"""
    ...
  def ggml_backend_buft_is_host(buft: ffi.CData) -> ffi.CData:
    """    GGML_API           bool                  ggml_backend_buft_is_host         (ggml_backend_buffer_type_t buft);"""
    ...
  def ggml_backend_buft_name(buft: ffi.CData) -> ffi.CData:
    """
    buffer type

        GGML_API           const char *          ggml_backend_buft_name            (ggml_backend_buffer_type_t buft);
    """
    ...
  def ggml_backend_buft_supports_backend(buft: ffi.CData, backend: ffi.CData) -> ffi.CData:
    """    GGML_API           bool                  ggml_backend_buft_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend);"""
    ...
  def ggml_backend_compare_graph_backend(backend1: ffi.CData, backend2: ffi.CData, graph: ffi.CData, callback: ffi.CData, user_data: ffi.CData) -> ffi.CData:
    """
    Compare the output of two backends

        GGML_API bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data);
    """
    ...
  def ggml_backend_cpu_buffer_from_ptr(ptr: ffi.CData, size: int) -> ffi.CData:
    """
    Create a backend buffer from an existing pointer

        GGML_API GGML_CALL ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size);
    """
    ...
  def ggml_backend_cpu_buffer_type() -> ffi.CData:
    """    GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void);"""
    ...
  def ggml_backend_cpu_init() -> ffi.CData:
    """    GGML_API ggml_backend_t ggml_backend_cpu_init(void);"""
    ...
  def ggml_backend_cpu_set_abort_callback(backend_cpu: ffi.CData, abort_callback: ffi.CData, abort_callback_data: ffi.CData) -> None:
    """    GGML_API           void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data);"""
    ...
  def ggml_backend_cpu_set_n_threads(backend_cpu: ffi.CData, n_threads: int) -> None:
    """    GGML_API           void ggml_backend_cpu_set_n_threads     (ggml_backend_t backend_cpu, int n_threads);"""
    ...
  def ggml_backend_cuda_buffer_type(device: int) -> ffi.CData:
    """
    device buffer

    GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);
    """
    ...
  def ggml_backend_cuda_get_device_count() -> int:
    """GGML_API GGML_CALL int  ggml_backend_cuda_get_device_count(void);"""
    ...
  def ggml_backend_cuda_get_device_description(device: int, description: ffi.CData, description_size: int) -> None:
    """GGML_API GGML_CALL void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);"""
    ...
  def ggml_backend_cuda_get_device_memory(device: int, free: ffi.CData, total: ffi.CData) -> None:
    """GGML_API GGML_CALL void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);"""
    ...
  def ggml_backend_cuda_host_buffer_type() -> ffi.CData:
    """
    pinned host buffer for use with the CPU backend for faster copies between CPU and GPU

    GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);
    """
    ...
  def ggml_backend_cuda_init(device: int) -> ffi.CData:
    """
    backend API

    GGML_API GGML_CALL ggml_backend_t ggml_backend_cuda_init(int device);
    """
    ...
  def ggml_backend_cuda_register_host_buffer(buffer: ffi.CData, size: int) -> ffi.CData:
    """GGML_API GGML_CALL bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);"""
    ...
  def ggml_backend_cuda_split_buffer_type(tensor_split: ffi.CData) -> ffi.CData:
    """
    split tensor buffer that splits matrices by rows across multiple devices

    GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(const float * tensor_split);
    """
    ...
  def ggml_backend_cuda_unregister_host_buffer(buffer: ffi.CData) -> None:
    """GGML_API GGML_CALL void ggml_backend_cuda_unregister_host_buffer(void * buffer);"""
    ...
  def ggml_backend_event_free(event: ffi.CData) -> None:
    """    GGML_API void                   ggml_backend_event_free       (ggml_backend_event_t event);"""
    ...
  def ggml_backend_event_new(backend: ffi.CData) -> ffi.CData:
    """
    events

        GGML_API ggml_backend_event_t   ggml_backend_event_new        (ggml_backend_t backend);
    """
    ...
  def ggml_backend_event_record(event: ffi.CData) -> None:
    """    GGML_API void                   ggml_backend_event_record     (ggml_backend_event_t event);"""
    ...
  def ggml_backend_event_synchronize(event: ffi.CData) -> None:
    """    GGML_API void                   ggml_backend_event_synchronize(ggml_backend_event_t event);"""
    ...
  def ggml_backend_event_wait(backend: ffi.CData, event: ffi.CData) -> None:
    """    GGML_API void                   ggml_backend_event_wait       (ggml_backend_t backend, ggml_backend_event_t event); // wait async on event"""
    ...
  def ggml_backend_free(backend: ffi.CData) -> None:
    """    GGML_API void         ggml_backend_free(ggml_backend_t backend);"""
    ...
  def ggml_backend_get_alignment(backend: ffi.CData) -> int:
    """    GGML_API size_t                     ggml_backend_get_alignment(ggml_backend_t backend);"""
    ...
  def ggml_backend_get_default_buffer_type(backend: ffi.CData) -> ffi.CData:
    """    GGML_API ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend);"""
    ...
  def ggml_backend_get_max_size(backend: ffi.CData) -> int:
    """    GGML_API size_t                     ggml_backend_get_max_size(ggml_backend_t backend);"""
    ...
  def ggml_backend_graph_compute(backend: ffi.CData, cgraph: ffi.CData) -> int:
    """    GGML_API enum ggml_status ggml_backend_graph_compute      (ggml_backend_t backend, struct ggml_cgraph * cgraph);"""
    ...
  def ggml_backend_graph_compute_async(backend: ffi.CData, cgraph: ffi.CData) -> int:
    """    GGML_API enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph);"""
    ...
  def ggml_backend_graph_copy(backend: ffi.CData, graph: ffi.CData) -> ffi.CData:
    """
    Copy a graph to a different backend

        GGML_API struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph);
    """
    ...
  def ggml_backend_graph_copy_free(copy: ffi.CData) -> None:
    """    GGML_API void                           ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy);"""
    ...
  def ggml_backend_graph_plan_compute(backend: ffi.CData, plan: ffi.CData) -> int:
    """    GGML_API enum ggml_status ggml_backend_graph_plan_compute (ggml_backend_t backend, ggml_backend_graph_plan_t plan);"""
    ...
  def ggml_backend_graph_plan_create(backend: ffi.CData, cgraph: ffi.CData) -> ffi.CData:
    """    GGML_API ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph);"""
    ...
  def ggml_backend_graph_plan_free(backend: ffi.CData, plan: ffi.CData) -> None:
    """    GGML_API void                      ggml_backend_graph_plan_free  (ggml_backend_t backend, ggml_backend_graph_plan_t plan);"""
    ...
  def ggml_backend_guid(backend: ffi.CData) -> ffi.CData:
    """    GGML_API ggml_guid_t  ggml_backend_guid(ggml_backend_t backend);"""
    ...
  def ggml_backend_is_cpu(backend: ffi.CData) -> ffi.CData:
    """    GGML_API GGML_CALL bool ggml_backend_is_cpu                (ggml_backend_t backend);"""
    ...
  def ggml_backend_is_cuda(backend: ffi.CData) -> ffi.CData:
    """GGML_API GGML_CALL bool ggml_backend_is_cuda(ggml_backend_t backend);"""
    ...
  def ggml_backend_is_kompute(backend: ffi.CData) -> ffi.CData:
    """GGML_API bool ggml_backend_is_kompute(ggml_backend_t backend);"""
    ...
  def ggml_backend_is_metal(backend: ffi.CData) -> ffi.CData:
    """GGML_API bool ggml_backend_is_metal(ggml_backend_t backend);"""
    ...
  def ggml_backend_is_rpc(backend: ffi.CData) -> ffi.CData:
    """GGML_API GGML_CALL bool ggml_backend_is_rpc(ggml_backend_t backend);"""
    ...
  def ggml_backend_is_vk(backend: ffi.CData) -> ffi.CData:
    """GGML_API GGML_CALL bool ggml_backend_is_vk(ggml_backend_t backend);"""
    ...
  def ggml_backend_kompute_buffer_type(device: int) -> ffi.CData:
    """GGML_API ggml_backend_buffer_type_t ggml_backend_kompute_buffer_type(int device);"""
    ...
  def ggml_backend_kompute_init(device: int) -> ffi.CData:
    """GGML_API ggml_backend_t ggml_backend_kompute_init(int device);"""
    ...
  def ggml_backend_metal_buffer_from_ptr(data: ffi.CData, size: int, max_size: int) -> ffi.CData:
    """GGML_API GGML_CALL ggml_backend_buffer_t ggml_backend_metal_buffer_from_ptr(void * data, size_t size, size_t max_size);"""
    ...
  def ggml_backend_metal_buffer_type() -> ffi.CData:
    """GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_metal_buffer_type(void);"""
    ...
  def ggml_backend_metal_capture_next_compute(backend: ffi.CData) -> None:
    """
    capture all command buffers committed the next time `ggml_backend_graph_compute` is called

    GGML_API void ggml_backend_metal_capture_next_compute(ggml_backend_t backend);
    """
    ...
  def ggml_backend_metal_init() -> ffi.CData:
    """GGML_API ggml_backend_t ggml_backend_metal_init(void);"""
    ...
  def ggml_backend_metal_log_set_callback(log_callback: ffi.CData, user_data: ffi.CData) -> None:
    """GGML_API void ggml_backend_metal_log_set_callback(ggml_log_callback log_callback, void * user_data);"""
    ...
  def ggml_backend_metal_set_n_cb(backend: ffi.CData, n_cb: int) -> None:
    """GGML_API void ggml_backend_metal_set_n_cb(ggml_backend_t backend, int n_cb);"""
    ...
  def ggml_backend_metal_supports_family(backend: ffi.CData, family: int) -> ffi.CData:
    """
    helper to check if the device supports a specific family
    ideally, the user code should be doing these checks
    ref: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf

    GGML_API bool ggml_backend_metal_supports_family(ggml_backend_t backend, int family);
    """
    ...
  def ggml_backend_name(backend: ffi.CData) -> ffi.CData:
    """    GGML_API const char * ggml_backend_name(ggml_backend_t backend);"""
    ...
  def ggml_backend_offload_op(backend: ffi.CData, op: ffi.CData) -> ffi.CData:
    """    GGML_API bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op);"""
    ...
  def ggml_backend_opencl_buffer_type() -> ffi.CData:
    """GGML_API ggml_backend_buffer_type_t ggml_backend_opencl_buffer_type(void);"""
    ...
  def ggml_backend_reg_alloc_buffer(i: int, size: int) -> ffi.CData:
    """    GGML_API ggml_backend_buffer_t      ggml_backend_reg_alloc_buffer(size_t i, size_t size);"""
    ...
  def ggml_backend_reg_find_by_name(name: ffi.CData) -> int:
    """    GGML_API size_t                     ggml_backend_reg_find_by_name(const char * name);"""
    ...
  def ggml_backend_reg_get_count() -> int:
    """    GGML_API size_t                     ggml_backend_reg_get_count(void);"""
    ...
  def ggml_backend_reg_get_default_buffer_type(i: int) -> ffi.CData:
    """    GGML_API ggml_backend_buffer_type_t ggml_backend_reg_get_default_buffer_type(size_t i);"""
    ...
  def ggml_backend_reg_get_name(i: int) -> ffi.CData:
    """    GGML_API const char *               ggml_backend_reg_get_name(size_t i);"""
    ...
  def ggml_backend_reg_init_backend(i: int, params: ffi.CData) -> ffi.CData:
    """    GGML_API ggml_backend_t             ggml_backend_reg_init_backend(size_t i, const char * params); // params is backend-specific"""
    ...
  def ggml_backend_reg_init_backend_from_str(backend_str: ffi.CData) -> ffi.CData:
    """    GGML_API ggml_backend_t             ggml_backend_reg_init_backend_from_str(const char * backend_str); // str is name[:params]"""
    ...
  def ggml_backend_rpc_buffer_type(endpoint: ffi.CData) -> ffi.CData:
    """GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint);"""
    ...
  def ggml_backend_rpc_get_device_memory(endpoint: ffi.CData, free: ffi.CData, total: ffi.CData) -> None:
    """GGML_API GGML_CALL void ggml_backend_rpc_get_device_memory(const char * endpoint, size_t * free, size_t * total);"""
    ...
  def ggml_backend_rpc_init(endpoint: ffi.CData) -> ffi.CData:
    """
    backend API

    GGML_API GGML_CALL ggml_backend_t ggml_backend_rpc_init(const char * endpoint);
    """
    ...
  def ggml_backend_sched_alloc_graph(sched: ffi.CData, graph: ffi.CData) -> ffi.CData:
    """
    Allocate and compute graph on the backend scheduler

        GGML_API bool                 ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
    """
    ...
  def ggml_backend_sched_free(sched: ffi.CData) -> None:
    """    GGML_API void                 ggml_backend_sched_free(ggml_backend_sched_t sched);"""
    ...
  def ggml_backend_sched_get_buffer_size(sched: ffi.CData, backend: ffi.CData) -> int:
    """    GGML_API size_t               ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend);"""
    ...
  def ggml_backend_sched_get_n_copies(sched: ffi.CData) -> int:
    """    GGML_API int                  ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched);"""
    ...
  def ggml_backend_sched_get_n_splits(sched: ffi.CData) -> int:
    """
    Get the number of splits of the last graph

        GGML_API int                  ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched);
    """
    ...
  def ggml_backend_sched_get_tensor_backend(sched: ffi.CData, node: ffi.CData) -> ffi.CData:
    """    GGML_API ggml_backend_t       ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node);"""
    ...
  def ggml_backend_sched_graph_compute(sched: ffi.CData, graph: ffi.CData) -> int:
    """    GGML_API enum ggml_status     ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph);"""
    ...
  def ggml_backend_sched_graph_compute_async(sched: ffi.CData, graph: ffi.CData) -> int:
    """    GGML_API enum ggml_status     ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph);"""
    ...
  def ggml_backend_sched_new(backends: ffi.CData, bufts: ffi.CData, n_backends: int, graph_size: int, parallel: ffi.CData) -> ffi.CData:
    """
    Initialize a backend scheduler

        GGML_API ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t * backends, ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size, bool parallel);
    """
    ...
  def ggml_backend_sched_reserve(sched: ffi.CData, measure_graph: ffi.CData) -> ffi.CData:
    """
    Initialize backend buffers from a measure graph

        GGML_API bool                 ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph);
    """
    ...
  def ggml_backend_sched_reset(sched: ffi.CData) -> None:
    """
    Reset all assignments and allocators - must be called before changing the node backends

        GGML_API void                 ggml_backend_sched_reset(ggml_backend_sched_t sched);
    """
    ...
  def ggml_backend_sched_set_eval_callback(sched: ffi.CData, callback: ffi.CData, user_data: ffi.CData) -> None:
    """
    Set a callback to be called for each resulting node during graph compute

        GGML_API void                 ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data);
    """
    ...
  def ggml_backend_sched_set_tensor_backend(sched: ffi.CData, node: ffi.CData, backend: ffi.CData) -> None:
    """    GGML_API void                 ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend);"""
    ...
  def ggml_backend_sched_synchronize(sched: ffi.CData) -> None:
    """    GGML_API void                 ggml_backend_sched_synchronize(ggml_backend_sched_t sched);"""
    ...
  def ggml_backend_supports_op(backend: ffi.CData, op: ffi.CData) -> ffi.CData:
    """    GGML_API bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op);"""
    ...
  def ggml_backend_sycl_buffer_type(device: int) -> ffi.CData:
    """
    devide buffer

    GGML_API ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(int device);
    """
    ...
  def ggml_backend_sycl_get_device_id(device_index: int) -> int:
    """
    TODO: these are temporary
    ref: https://github.com/ggerganov/llama.cpp/pull/6022#issuecomment-1992615670

    GGML_API GGML_CALL int ggml_backend_sycl_get_device_id(int device_index);
    """
    ...
  def ggml_backend_sycl_get_device_index(device_id: int) -> int:
    """GGML_API GGML_CALL int ggml_backend_sycl_get_device_index(int device_id);"""
    ...
  def ggml_backend_sycl_get_device_memory(device: int, free: ffi.CData, total: ffi.CData) -> None:
    """GGML_API GGML_CALL void ggml_backend_sycl_get_device_memory(int device, size_t *free, size_t *total);"""
    ...
  def ggml_backend_sycl_host_buffer_type() -> ffi.CData:
    """
    pinned host buffer for use with the CPU backend for faster copies between CPU and GPU

    GGML_API ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type(void);
    """
    ...
  def ggml_backend_sycl_init(device: int) -> ffi.CData:
    """
    backend API

    GGML_API ggml_backend_t ggml_backend_sycl_init(int device);
    """
    ...
  def ggml_backend_sycl_print_sycl_devices() -> None:
    """GGML_API void   ggml_backend_sycl_print_sycl_devices(void);"""
    ...
  def ggml_backend_sycl_set_single_device_mode(main_gpu_id: int) -> None:
    """GGML_API GGML_CALL void ggml_backend_sycl_set_single_device_mode(int main_gpu_id);"""
    ...
  def ggml_backend_sycl_split_buffer_type(tensor_split: ffi.CData) -> ffi.CData:
    """
    split tensor buffer that splits matrices by rows across multiple devices

    GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(const float * tensor_split);
    """
    ...
  def ggml_backend_synchronize(backend: ffi.CData) -> None:
    """    GGML_API void ggml_backend_synchronize(ggml_backend_t backend);"""
    ...
  def ggml_backend_tensor_alloc(buffer: ffi.CData, tensor: ffi.CData, addr: ffi.CData) -> None:
    """
    Tensor initialization

        GGML_API void ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr);
    """
    ...
  def ggml_backend_tensor_copy(src: ffi.CData, dst: ffi.CData) -> None:
    """
    tensor copy between different backends

        GGML_API void ggml_backend_tensor_copy(struct ggml_tensor * src, struct ggml_tensor * dst);
    """
    ...
  def ggml_backend_tensor_copy_async(backend_src: ffi.CData, backend_dst: ffi.CData, src: ffi.CData, dst: ffi.CData) -> None:
    """
    asynchronous copy
    the copy is performed after all the currently queued operations in backend_src
    backend_dst will wait for the copy to complete before performing other operations
    automatic fallback to sync copy if async is not supported

        GGML_API void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, struct ggml_tensor * src, struct ggml_tensor * dst);
    """
    ...
  def ggml_backend_tensor_get(tensor: ffi.CData, data: ffi.CData, offset: int, size: int) -> None:
    """    GGML_API GGML_CALL void ggml_backend_tensor_get(const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);"""
    ...
  def ggml_backend_tensor_get_async(backend: ffi.CData, tensor: ffi.CData, data: ffi.CData, offset: int, size: int) -> None:
    """    GGML_API void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);"""
    ...
  def ggml_backend_tensor_set(tensor: ffi.CData, data: ffi.CData, offset: int, size: int) -> None:
    """    GGML_API GGML_CALL void ggml_backend_tensor_set(      struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);"""
    ...
  def ggml_backend_tensor_set_async(backend: ffi.CData, tensor: ffi.CData, data: ffi.CData, offset: int, size: int) -> None:
    """    GGML_API void ggml_backend_tensor_set_async(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);"""
    ...
  def ggml_backend_view_init(buffer: ffi.CData, tensor: ffi.CData) -> None:
    """    GGML_API void ggml_backend_view_init(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);"""
    ...
  def ggml_backend_vk_buffer_type(dev_num: int) -> ffi.CData:
    """GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);"""
    ...
  def ggml_backend_vk_get_device_count() -> int:
    """GGML_API GGML_CALL int  ggml_backend_vk_get_device_count(void);"""
    ...
  def ggml_backend_vk_get_device_description(device: int, description: ffi.CData, description_size: int) -> None:
    """GGML_API GGML_CALL void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);"""
    ...
  def ggml_backend_vk_get_device_memory(device: int, free: ffi.CData, total: ffi.CData) -> None:
    """GGML_API GGML_CALL void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);"""
    ...
  def ggml_backend_vk_host_buffer_type() -> ffi.CData:
    """
    pinned host buffer for use with the CPU backend for faster copies between CPU and GPU

    GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);
    """
    ...
  def ggml_backend_vk_init(dev_num: int) -> ffi.CData:
    """
    backend API

    GGML_API GGML_CALL ggml_backend_t ggml_backend_vk_init(size_t dev_num);
    """
    ...
  def ggml_bf16_to_fp32(arg: ffi.CData) -> float:
    """    GGML_API float       ggml_bf16_to_fp32(ggml_bf16_t);  // consider just doing << 16"""
    ...
  def ggml_bf16_to_fp32_row(arg: ffi.CData, arg2: ffi.CData, arg3: int) -> None:
    """    GGML_API void        ggml_bf16_to_fp32_row(const ggml_bf16_t *, float *, int64_t);"""
    ...
  def ggml_blck_size(type: int) -> int:
    """    GGML_API GGML_CALL int    ggml_blck_size(enum ggml_type type);"""
    ...
  def ggml_build_backward_expand(ctx: ffi.CData, gf: ffi.CData, gb: ffi.CData, keep: ffi.CData) -> None:
    """    GGML_API void ggml_build_backward_expand(struct ggml_context * ctx, struct ggml_cgraph * gf, struct ggml_cgraph * gb, bool keep);"""
    ...
  def ggml_build_backward_gradient_checkpointing(ctx: ffi.CData, gf: ffi.CData, gb: ffi.CData, gb_tmp: ffi.CData, checkpoints: ffi.CData, n_checkpoints: int) -> None:
    """
    build gradient checkpointing backward graph gb for gf using provided checkpoints
    gb_tmp will contain original backward graph with rewritten backward process nodes,
    but without the second forward pass nodes.

        GGML_API void ggml_build_backward_gradient_checkpointing(
                struct ggml_context   * ctx,
                struct ggml_cgraph    * gf,
                struct ggml_cgraph    * gb,
                struct ggml_cgraph    * gb_tmp,
                struct ggml_tensor  * * checkpoints,
                int                     n_checkpoints);
    """
    ...
  def ggml_build_forward_expand(cgraph: ffi.CData, tensor: ffi.CData) -> None:
    """    GGML_API void ggml_build_forward_expand (struct ggml_cgraph * cgraph, struct ggml_tensor * tensor);"""
    ...
  def ggml_cast(ctx: ffi.CData, a: ffi.CData, type: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_cast(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                enum   ggml_type      type);
    """
    ...
  def ggml_cl_add(src0: ffi.CData, src1: ffi.CData, dst: ffi.CData) -> None:
    """GGML_API void   ggml_cl_add(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);"""
    ...
  def ggml_cl_can_mul_mat(src0: ffi.CData, src1: ffi.CData, dst: ffi.CData) -> ffi.CData:
    """GGML_API bool   ggml_cl_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst);"""
    ...
  def ggml_cl_free_data(tensor: ffi.CData) -> None:
    """GGML_API void ggml_cl_free_data(const struct ggml_tensor* tensor);"""
    ...
  def ggml_cl_init() -> None:
    """GGML_API void ggml_cl_init(void);"""
    ...
  def ggml_cl_mul(src0: ffi.CData, src1: ffi.CData, dst: ffi.CData) -> None:
    """GGML_API void   ggml_cl_mul(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);"""
    ...
  def ggml_cl_mul_mat(src0: ffi.CData, src1: ffi.CData, dst: ffi.CData, wdata: ffi.CData, wsize: int) -> None:
    """GGML_API void   ggml_cl_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, void * wdata, size_t wsize);"""
    ...
  def ggml_cl_mul_mat_get_wsize(src0: ffi.CData, src1: ffi.CData, dst: ffi.CData) -> int:
    """GGML_API size_t ggml_cl_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);"""
    ...
  def ggml_cl_transform_tensor(data: ffi.CData, tensor: ffi.CData) -> None:
    """GGML_API void ggml_cl_transform_tensor(void * data, struct ggml_tensor * tensor);"""
    ...
  def ggml_clamp(ctx: ffi.CData, a: ffi.CData, min: float, max: float) -> ffi.CData:
    """
    clamp
    in-place, returns view(a)

        GGML_API struct ggml_tensor * ggml_clamp(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                float                 min,
                float                 max);
    """
    ...
  def ggml_concat(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
    concat a and b on dim 2
    used in stable-diffusion

        GGML_API struct ggml_tensor * ggml_concat(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_cont(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
    make contiguous

        GGML_API struct ggml_tensor * ggml_cont(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_cont_1d(ctx: ffi.CData, a: ffi.CData, ne0: int) -> ffi.CData:
    """
    make contiguous, with new shape

        GGML_API struct ggml_tensor * ggml_cont_1d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int64_t               ne0);
    """
    ...
  def ggml_cont_2d(ctx: ffi.CData, a: ffi.CData, ne0: int, ne1: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_cont_2d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int64_t               ne0,
                int64_t               ne1);
    """
    ...
  def ggml_cont_3d(ctx: ffi.CData, a: ffi.CData, ne0: int, ne1: int, ne2: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_cont_3d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int64_t               ne0,
                int64_t               ne1,
                int64_t               ne2);
    """
    ...
  def ggml_cont_4d(ctx: ffi.CData, a: ffi.CData, ne0: int, ne1: int, ne2: int, ne3: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_cont_4d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int64_t               ne0,
                int64_t               ne1,
                int64_t               ne2,
                int64_t               ne3);
    """
    ...
  def ggml_conv_1d(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, s0: int, p0: int, d0: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_conv_1d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                int                   s0,  // stride
                int                   p0,  // padding
                int                   d0); // dilation
    """
    ...
  def ggml_conv_1d_ph(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, s: int, d: int) -> ffi.CData:
    """
    conv_1d with padding = half
    alias for ggml_conv_1d(a, b, s, a->ne[0]/2, d)

        GGML_API struct ggml_tensor* ggml_conv_1d_ph(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                int                   s,
                int                   d);
    """
    ...
  def ggml_conv_2d(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, s0: int, s1: int, p0: int, p1: int, d0: int, d1: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_conv_2d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                int                   s0,
                int                   s1,
                int                   p0,
                int                   p1,
                int                   d0,
                int                   d1);
    """
    ...
  def ggml_conv_2d_s1_ph(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
    kernel size is a->ne[0] x a->ne[1]
    stride is 1
    padding is half
    example:
    a:      3    3    256  256
    b:     64   64    256    1
    res:   64   64    256    1
    used in sam

        GGML_API struct ggml_tensor * ggml_conv_2d_s1_ph(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_conv_2d_sk_p0(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
    kernel size is a->ne[0] x a->ne[1]
    stride is equal to kernel size
    padding is zero
    example:
    a:     16   16    3  768
    b:   1024 1024    3    1
    res:   64   64  768    1
    used in sam

        GGML_API struct ggml_tensor * ggml_conv_2d_sk_p0(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_conv_depthwise_2d(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, s0: int, s1: int, p0: int, p1: int, d0: int, d1: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_conv_depthwise_2d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                int                  s0,
                int                  s1,
                int                  p0,
                int                  p1,
                int                  d0,
                int                  d1);
    """
    ...
  def ggml_conv_transpose_1d(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, s0: int, p0: int, d0: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_conv_transpose_1d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                int                   s0,
                int                   p0,
                int                   d0);
    """
    ...
  def ggml_conv_transpose_2d_p0(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, stride: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_conv_transpose_2d_p0(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                int                   stride);
    """
    ...
  def ggml_cpu_has_arm_fma() -> int:
    """    GGML_API int ggml_cpu_has_arm_fma    (void);"""
    ...
  def ggml_cpu_has_avx() -> int:
    """    GGML_API int ggml_cpu_has_avx        (void);"""
    ...
  def ggml_cpu_has_avx2() -> int:
    """    GGML_API int ggml_cpu_has_avx2       (void);"""
    ...
  def ggml_cpu_has_avx512() -> int:
    """    GGML_API int ggml_cpu_has_avx512     (void);"""
    ...
  def ggml_cpu_has_avx512_vbmi() -> int:
    """    GGML_API int ggml_cpu_has_avx512_vbmi(void);"""
    ...
  def ggml_cpu_has_avx512_vnni() -> int:
    """    GGML_API int ggml_cpu_has_avx512_vnni(void);"""
    ...
  def ggml_cpu_has_avx_vnni() -> int:
    """    GGML_API int ggml_cpu_has_avx_vnni   (void);"""
    ...
  def ggml_cpu_has_blas() -> int:
    """    GGML_API int ggml_cpu_has_blas       (void);"""
    ...
  def ggml_cpu_has_clblast() -> int:
    """    GGML_API int ggml_cpu_has_clblast    (void);"""
    ...
  def ggml_cpu_has_cuda() -> int:
    """    GGML_API int ggml_cpu_has_cuda       (void);"""
    ...
  def ggml_cpu_has_f16c() -> int:
    """    GGML_API int ggml_cpu_has_f16c       (void);"""
    ...
  def ggml_cpu_has_fma() -> int:
    """    GGML_API int ggml_cpu_has_fma        (void);"""
    ...
  def ggml_cpu_has_fp16_va() -> int:
    """    GGML_API int ggml_cpu_has_fp16_va    (void);"""
    ...
  def ggml_cpu_has_gpublas() -> int:
    """    GGML_API int ggml_cpu_has_gpublas    (void);"""
    ...
  def ggml_cpu_has_kompute() -> int:
    """    GGML_API int ggml_cpu_has_kompute    (void);"""
    ...
  def ggml_cpu_has_matmul_int8() -> int:
    """    GGML_API int ggml_cpu_has_matmul_int8(void);"""
    ...
  def ggml_cpu_has_metal() -> int:
    """    GGML_API int ggml_cpu_has_metal      (void);"""
    ...
  def ggml_cpu_has_neon() -> int:
    """    GGML_API int ggml_cpu_has_neon       (void);"""
    ...
  def ggml_cpu_has_sse3() -> int:
    """    GGML_API int ggml_cpu_has_sse3       (void);"""
    ...
  def ggml_cpu_has_ssse3() -> int:
    """    GGML_API int ggml_cpu_has_ssse3      (void);"""
    ...
  def ggml_cpu_has_sycl() -> int:
    """    GGML_API int ggml_cpu_has_sycl       (void);"""
    ...
  def ggml_cpu_has_vsx() -> int:
    """    GGML_API int ggml_cpu_has_vsx        (void);"""
    ...
  def ggml_cpu_has_vulkan() -> int:
    """    GGML_API int ggml_cpu_has_vulkan     (void);"""
    ...
  def ggml_cpu_has_wasm_simd() -> int:
    """    GGML_API int ggml_cpu_has_wasm_simd  (void);"""
    ...
  def ggml_cpy(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
    a -> b, return view(b)

        GGML_API struct ggml_tensor * ggml_cpy(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_cross_entropy_loss(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_cross_entropy_loss(
                struct ggml_context         * ctx,
                struct ggml_tensor          * a,
                struct ggml_tensor          * b);
    """
    ...
  def ggml_cross_entropy_loss_back(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, c: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_cross_entropy_loss_back(
                struct ggml_context         * ctx,
                struct ggml_tensor          * a,
                struct ggml_tensor          * b,
                struct ggml_tensor          * c);
    """
    ...
  def ggml_cycles() -> int:
    """    GGML_API int64_t ggml_cycles(void);"""
    ...
  def ggml_cycles_per_ms() -> int:
    """    GGML_API int64_t ggml_cycles_per_ms(void);"""
    ...
  def ggml_diag(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_diag(
            struct ggml_context     * ctx,
            struct ggml_tensor      * a);
    """
    ...
  def ggml_diag_mask_inf(ctx: ffi.CData, a: ffi.CData, n_past: int) -> ffi.CData:
    """
    set elements above the diagonal to -INF

        GGML_API struct ggml_tensor * ggml_diag_mask_inf(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   n_past);
    """
    ...
  def ggml_diag_mask_inf_inplace(ctx: ffi.CData, a: ffi.CData, n_past: int) -> ffi.CData:
    """
    in-place, returns view(a)

        GGML_API struct ggml_tensor * ggml_diag_mask_inf_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   n_past);
    """
    ...
  def ggml_diag_mask_zero(ctx: ffi.CData, a: ffi.CData, n_past: int) -> ffi.CData:
    """
    set elements above the diagonal to 0

        GGML_API struct ggml_tensor * ggml_diag_mask_zero(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   n_past);
    """
    ...
  def ggml_diag_mask_zero_inplace(ctx: ffi.CData, a: ffi.CData, n_past: int) -> ffi.CData:
    """
    in-place, returns view(a)

        GGML_API struct ggml_tensor * ggml_diag_mask_zero_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   n_past);
    """
    ...
  def ggml_div(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_div(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_div_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_div_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_dup(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_dup(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_dup_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
    in-place, returns view(a)

        GGML_API struct ggml_tensor * ggml_dup_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_dup_tensor(ctx: ffi.CData, src: ffi.CData) -> ffi.CData:
    """    GGML_API struct ggml_tensor * ggml_dup_tensor (struct ggml_context * ctx, const struct ggml_tensor * src);"""
    ...
  def ggml_element_size(tensor: ffi.CData) -> int:
    """    GGML_API GGML_CALL size_t  ggml_element_size(const struct ggml_tensor * tensor);"""
    ...
  def ggml_elu(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_elu(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_elu_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_elu_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_flash_attn(ctx: ffi.CData, q: ffi.CData, k: ffi.CData, v: ffi.CData, masked: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_flash_attn(
                struct ggml_context * ctx,
                struct ggml_tensor  * q,
                struct ggml_tensor  * k,
                struct ggml_tensor  * v,
                bool                  masked);
    """
    ...
  def ggml_flash_attn_back(ctx: ffi.CData, q: ffi.CData, k: ffi.CData, v: ffi.CData, d: ffi.CData, masked: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_flash_attn_back(
               struct ggml_context * ctx,
               struct ggml_tensor  * q,
               struct ggml_tensor  * k,
               struct ggml_tensor  * v,
               struct ggml_tensor  * d,
               bool                  masked);
    """
    ...
  def ggml_flash_attn_ext(ctx: ffi.CData, q: ffi.CData, k: ffi.CData, v: ffi.CData, mask: ffi.CData, scale: float, max_bias: float) -> ffi.CData:
    """
    q:    [n_embd, n_batch,     n_head,    1]
    k:    [n_embd, n_kv,        n_head_kv, 1]
    v:    [n_embd, n_kv,        n_head_kv, 1] !! not transposed !!
    mask: [n_kv,   n_batch_pad, 1,         1] !! n_batch_pad = GGML_PAD(n_batch, GGML_KQ_MASK_PAD) !!
    res:  [n_embd, n_head,      n_batch,   1] !! permuted !!

        GGML_API struct ggml_tensor * ggml_flash_attn_ext(
                struct ggml_context * ctx,
                struct ggml_tensor  * q,
                struct ggml_tensor  * k,
                struct ggml_tensor  * v,
                struct ggml_tensor  * mask,
                float                 scale,
                float                 max_bias);
    """
    ...
  def ggml_flash_attn_ext_set_prec(a: ffi.CData, prec: int) -> None:
    """
        GGML_API void ggml_flash_attn_ext_set_prec(
                struct ggml_tensor * a,
                enum ggml_prec       prec);
    """
    ...
  def ggml_flash_ff(ctx: ffi.CData, a: ffi.CData, b0: ffi.CData, b1: ffi.CData, c0: ffi.CData, c1: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_flash_ff(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b0,
                struct ggml_tensor  * b1,
                struct ggml_tensor  * c0,
                struct ggml_tensor  * c1);
    """
    ...
  def ggml_fopen(fname: ffi.CData, mode: ffi.CData) -> ffi.CData:
    """
    accepts a UTF-8 path, even on Windows

        GGML_API FILE *  ggml_fopen(const char * fname, const char * mode);
    """
    ...
  def ggml_format_name(tensor: ffi.CData, fmt: ffi.CData, *args2) -> ffi.CData:
    """    GGML_API struct ggml_tensor * ggml_format_name(      struct ggml_tensor * tensor, const char * fmt, ...);"""
    ...
  def ggml_fp16_to_fp32(arg: np.float16) -> float:
    """    GGML_API float       ggml_fp16_to_fp32(ggml_fp16_t);"""
    ...
  def ggml_fp16_to_fp32_row(arg: ffi.CData, arg2: ffi.CData, arg3: int) -> None:
    """    GGML_API void        ggml_fp16_to_fp32_row(const ggml_fp16_t *, float *, int64_t);"""
    ...
  def ggml_fp32_to_bf16(arg: float) -> ffi.CData:
    """    GGML_API ggml_bf16_t ggml_fp32_to_bf16(float);"""
    ...
  def ggml_fp32_to_bf16_row(arg: ffi.CData, arg2: ffi.CData, arg3: int) -> None:
    """    GGML_API void        ggml_fp32_to_bf16_row(const float *, ggml_bf16_t *, int64_t);"""
    ...
  def ggml_fp32_to_fp16(arg: float) -> np.float16:
    """    GGML_API ggml_fp16_t ggml_fp32_to_fp16(float);"""
    ...
  def ggml_fp32_to_fp16_row(arg: ffi.CData, arg2: ffi.CData, arg3: int) -> None:
    """    GGML_API void        ggml_fp32_to_fp16_row(const float *, ggml_fp16_t *, int64_t);"""
    ...
  def ggml_free(ctx: ffi.CData) -> None:
    """    GGML_API void                  ggml_free(struct ggml_context * ctx);"""
    ...
  def ggml_ftype_to_ggml_type(ftype: int) -> int:
    """
    TODO: temporary until model loading of ggml examples is refactored

        GGML_API enum ggml_type ggml_ftype_to_ggml_type(enum ggml_ftype ftype);
    """
    ...
  def ggml_gallocr_alloc_graph(galloc: ffi.CData, graph: ffi.CData) -> ffi.CData:
    """
    automatic reallocation if the topology changes when using a single buffer
    returns false if using multiple buffers and a re-allocation is needed (call ggml_gallocr_reserve_n first to set the node buffers)

    GGML_API bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
    """
    ...
  def ggml_gallocr_free(galloc: ffi.CData) -> None:
    """GGML_API void           ggml_gallocr_free(ggml_gallocr_t galloc);"""
    ...
  def ggml_gallocr_get_buffer_size(galloc: ffi.CData, buffer_id: int) -> int:
    """GGML_API size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id);"""
    ...
  def ggml_gallocr_new(buft: ffi.CData) -> ffi.CData:
    """GGML_API ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft);"""
    ...
  def ggml_gallocr_new_n(bufts: ffi.CData, n_bufs: int) -> ffi.CData:
    """GGML_API ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs);"""
    ...
  def ggml_gallocr_reserve(galloc: ffi.CData, graph: ffi.CData) -> ffi.CData:
    """
    pre-allocate buffers from a measure graph - does not allocate or modify the graph
    call with a worst-case graph to avoid buffer reallocations
    not strictly required for single buffer usage: ggml_gallocr_alloc_graph will reallocate the buffers automatically if needed
    returns false if the buffer allocation failed

    GGML_API bool ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
    """
    ...
  def ggml_gallocr_reserve_n(galloc: ffi.CData, graph: ffi.CData, node_buffer_ids: ffi.CData, leaf_buffer_ids: ffi.CData) -> ffi.CData:
    """
    GGML_API bool ggml_gallocr_reserve_n(
        ggml_gallocr_t galloc,
        struct ggml_cgraph * graph,
        const int * node_buffer_ids,
        const int * leaf_buffer_ids);
    """
    ...
  def ggml_gelu(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_gelu(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_gelu_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_gelu_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_gelu_quick(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_gelu_quick(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_gelu_quick_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_gelu_quick_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_get_data(tensor: ffi.CData) -> ffi.CData:
    """    GGML_API void *  ggml_get_data    (const struct ggml_tensor * tensor);"""
    ...
  def ggml_get_data_f32(tensor: ffi.CData) -> ffi.CData:
    """    GGML_API float * ggml_get_data_f32(const struct ggml_tensor * tensor);"""
    ...
  def ggml_get_f32_1d(tensor: ffi.CData, i: int) -> float:
    """    GGML_API float   ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);"""
    ...
  def ggml_get_f32_nd(tensor: ffi.CData, i0: int, i1: int, i2: int, i3: int) -> float:
    """    GGML_API float   ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);"""
    ...
  def ggml_get_first_tensor(ctx: ffi.CData) -> ffi.CData:
    """
    Context tensor enumeration and lookup

        GGML_API struct ggml_tensor * ggml_get_first_tensor(const struct ggml_context * ctx);
    """
    ...
  def ggml_get_i32_1d(tensor: ffi.CData, i: int) -> int:
    """    GGML_API int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);"""
    ...
  def ggml_get_i32_nd(tensor: ffi.CData, i0: int, i1: int, i2: int, i3: int) -> int:
    """    GGML_API int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);"""
    ...
  def ggml_get_max_tensor_size(ctx: ffi.CData) -> int:
    """    GGML_API size_t  ggml_get_max_tensor_size(const struct ggml_context * ctx);"""
    ...
  def ggml_get_mem_buffer(ctx: ffi.CData) -> ffi.CData:
    """    GGML_API void *  ggml_get_mem_buffer     (const struct ggml_context * ctx);"""
    ...
  def ggml_get_mem_size(ctx: ffi.CData) -> int:
    """    GGML_API size_t  ggml_get_mem_size       (const struct ggml_context * ctx);"""
    ...
  def ggml_get_name(tensor: ffi.CData) -> ffi.CData:
    """    GGML_API const char *         ggml_get_name   (const struct ggml_tensor * tensor);"""
    ...
  def ggml_get_next_tensor(ctx: ffi.CData, tensor: ffi.CData) -> ffi.CData:
    """    GGML_API struct ggml_tensor * ggml_get_next_tensor (const struct ggml_context * ctx, struct ggml_tensor * tensor);"""
    ...
  def ggml_get_no_alloc(ctx: ffi.CData) -> ffi.CData:
    """    GGML_API bool    ggml_get_no_alloc(struct ggml_context * ctx);"""
    ...
  def ggml_get_rel_pos(ctx: ffi.CData, a: ffi.CData, qh: int, kh: int) -> ffi.CData:
    """
    used in sam

        GGML_API struct ggml_tensor * ggml_get_rel_pos(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   qh,
                int                   kh);
    """
    ...
  def ggml_get_rows(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
    supports 3D: a->ne[2] == b->ne[1]

        GGML_API struct ggml_tensor * ggml_get_rows(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_get_rows_back(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, c: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_get_rows_back(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                struct ggml_tensor  * c);
    """
    ...
  def ggml_get_tensor(ctx: ffi.CData, name: ffi.CData) -> ffi.CData:
    """    GGML_API struct ggml_tensor * ggml_get_tensor(struct ggml_context * ctx, const char * name);"""
    ...
  def ggml_get_unary_op(tensor: ffi.CData) -> int:
    """    GGML_API GGML_CALL enum ggml_unary_op ggml_get_unary_op(const struct ggml_tensor * tensor);"""
    ...
  def ggml_graph_clear(cgraph: ffi.CData) -> None:
    """    GGML_API void                 ggml_graph_clear       (struct ggml_cgraph * cgraph);"""
    ...
  def ggml_graph_compute(cgraph: ffi.CData, cplan: ffi.CData) -> int:
    """    GGML_API enum ggml_status  ggml_graph_compute         (      struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);"""
    ...
  def ggml_graph_compute_with_ctx(ctx: ffi.CData, cgraph: ffi.CData, n_threads: int) -> int:
    """
    same as ggml_graph_compute() but the work data is allocated as a part of the context
    note: the drawback of this API is that you must have ensured that the context has enough memory for the work data

        GGML_API enum ggml_status  ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);
    """
    ...
  def ggml_graph_cpy(src: ffi.CData, dst: ffi.CData) -> None:
    """    GGML_API void                 ggml_graph_cpy         (struct ggml_cgraph * src, struct ggml_cgraph * dst);"""
    ...
  def ggml_graph_dump_dot(gb: ffi.CData, gf: ffi.CData, filename: ffi.CData) -> None:
    """
    dump the graph into a file using the dot format

        GGML_API void ggml_graph_dump_dot(const struct ggml_cgraph * gb, const struct ggml_cgraph * gf, const char * filename);
    """
    ...
  def ggml_graph_dup(ctx: ffi.CData, cgraph: ffi.CData) -> ffi.CData:
    """    GGML_API struct ggml_cgraph * ggml_graph_dup         (struct ggml_context * ctx, struct ggml_cgraph * cgraph);"""
    ...
  def ggml_graph_export(cgraph: ffi.CData, fname: ffi.CData) -> None:
    """    GGML_API void                 ggml_graph_export(const struct ggml_cgraph * cgraph, const char * fname);"""
    ...
  def ggml_graph_get_tensor(cgraph: ffi.CData, name: ffi.CData) -> ffi.CData:
    """    GGML_API struct ggml_tensor * ggml_graph_get_tensor(struct ggml_cgraph * cgraph, const char * name);"""
    ...
  def ggml_graph_import(fname: ffi.CData, ctx_data: ffi.CData, ctx_eval: ffi.CData) -> ffi.CData:
    """    GGML_API struct ggml_cgraph * ggml_graph_import(const char * fname, struct ggml_context ** ctx_data, struct ggml_context ** ctx_eval);"""
    ...
  def ggml_graph_overhead() -> int:
    """    GGML_API size_t ggml_graph_overhead(void);"""
    ...
  def ggml_graph_overhead_custom(size: int, grads: ffi.CData) -> int:
    """    GGML_API size_t ggml_graph_overhead_custom(size_t size, bool grads);"""
    ...
  def ggml_graph_plan(cgraph: ffi.CData, n_threads: int) -> ffi.CData:
    """
    ggml_graph_plan() has to be called before ggml_graph_compute()
    when plan.work_size > 0, caller must allocate memory for plan.work_data

        GGML_API struct ggml_cplan ggml_graph_plan            (const struct ggml_cgraph * cgraph, int n_threads /*= GGML_DEFAULT_N_THREADS*/);
    """
    ...
  def ggml_graph_print(cgraph: ffi.CData) -> None:
    """
    print info and performance information for the graph

        GGML_API void ggml_graph_print(const struct ggml_cgraph * cgraph);
    """
    ...
  def ggml_graph_reset(cgraph: ffi.CData) -> None:
    """    GGML_API void                 ggml_graph_reset       (struct ggml_cgraph * cgraph);  // zero grads"""
    ...
  def ggml_graph_view(cgraph: ffi.CData, i0: int, i1: int) -> ffi.CData:
    """    GGML_API struct ggml_cgraph   ggml_graph_view        (struct ggml_cgraph * cgraph, int i0, int i1);"""
    ...
  def ggml_group_norm(ctx: ffi.CData, a: ffi.CData, n_groups: int) -> ffi.CData:
    """
    group normalize along ne0*ne1*n_groups
    used in stable-diffusion
    TODO: eps is hardcoded to 1e-6 for now

        GGML_API struct ggml_tensor * ggml_group_norm(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   n_groups);
    """
    ...
  def ggml_group_norm_inplace(ctx: ffi.CData, a: ffi.CData, n_groups: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_group_norm_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   n_groups);
    """
    ...
  def ggml_guid_matches(guid_a: ffi.CData, guid_b: ffi.CData) -> ffi.CData:
    """    GGML_API bool ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b);"""
    ...
  def ggml_hardsigmoid(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
    hardsigmoid(x) = relu6(x + 3) / 6

        GGML_API struct ggml_tensor * ggml_hardsigmoid(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_hardswish(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
    hardswish(x) = x * relu6(x + 3) / 6

        GGML_API struct ggml_tensor * ggml_hardswish(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_im2col(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, s0: int, s1: int, p0: int, p1: int, d0: int, d1: int, is_2D: ffi.CData, dst_type: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_im2col(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                int                  s0,
                int                  s1,
                int                  p0,
                int                  p1,
                int                  d0,
                int                  d1,
                bool                 is_2D,
                enum ggml_type       dst_type);
    """
    ...
  def ggml_init(params: ffi.CData) -> ffi.CData:
    """    GGML_API struct ggml_context * ggml_init(struct ggml_init_params params);"""
    ...
  def ggml_internal_get_type_traits(type: int) -> ffi.CData:
    """    GGML_API ggml_type_traits_t ggml_internal_get_type_traits(enum ggml_type type);"""
    ...
  def ggml_is_3d(tensor: ffi.CData) -> ffi.CData:
    """    GGML_API           bool ggml_is_3d        (const struct ggml_tensor * tensor);"""
    ...
  def ggml_is_contiguous(tensor: ffi.CData) -> ffi.CData:
    """    GGML_API GGML_CALL bool ggml_is_contiguous(const struct ggml_tensor * tensor);"""
    ...
  def ggml_is_empty(tensor: ffi.CData) -> ffi.CData:
    """    GGML_API GGML_CALL bool ggml_is_empty     (const struct ggml_tensor * tensor);"""
    ...
  def ggml_is_matrix(tensor: ffi.CData) -> ffi.CData:
    """    GGML_API           bool ggml_is_matrix    (const struct ggml_tensor * tensor);"""
    ...
  def ggml_is_numa() -> ffi.CData:
    """    GGML_API bool    ggml_is_numa(void); // true if init detected that system has >1 NUMA node"""
    ...
  def ggml_is_permuted(tensor: ffi.CData) -> ffi.CData:
    """    GGML_API GGML_CALL bool ggml_is_permuted  (const struct ggml_tensor * tensor);"""
    ...
  def ggml_is_quantized(type: int) -> ffi.CData:
    """    GGML_API GGML_CALL bool    ggml_is_quantized(enum ggml_type type);"""
    ...
  def ggml_is_scalar(tensor: ffi.CData) -> ffi.CData:
    """    GGML_API           bool ggml_is_scalar    (const struct ggml_tensor * tensor);"""
    ...
  def ggml_is_transposed(tensor: ffi.CData) -> ffi.CData:
    """    GGML_API GGML_CALL bool ggml_is_transposed(const struct ggml_tensor * tensor);"""
    ...
  def ggml_is_vector(tensor: ffi.CData) -> ffi.CData:
    """    GGML_API           bool ggml_is_vector    (const struct ggml_tensor * tensor);"""
    ...
  def ggml_leaky_relu(ctx: ffi.CData, a: ffi.CData, negative_slope: float, inplace: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_leaky_relu(
                struct ggml_context * ctx,
                struct ggml_tensor  * a, float negative_slope, bool inplace);
    """
    ...
  def ggml_log(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_log(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_log_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_log_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_map_binary_f32(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_binary_f32(
                struct ggml_context         * ctx,
                struct ggml_tensor          * a,
                struct ggml_tensor          * b,
                       ggml_binary_op_f32_t   fun),
            "use ggml_map_custom2 instead");
    """
    ...
  def ggml_map_binary_inplace_f32(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_binary_inplace_f32(
                struct ggml_context         * ctx,
                struct ggml_tensor          * a,
                struct ggml_tensor          * b,
                       ggml_binary_op_f32_t   fun),
            "use ggml_map_custom2_inplace instead");
    """
    ...
  def ggml_map_custom1(ctx: ffi.CData, a: ffi.CData, fun: ffi.CData, n_tasks: int, userdata: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_map_custom1(
                struct ggml_context   * ctx,
                struct ggml_tensor    * a,
                ggml_custom1_op_t       fun,
                int                     n_tasks,
                void                  * userdata);
    """
    ...
  def ggml_map_custom1_f32(ctx: ffi.CData, a: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_custom1_f32(
                struct ggml_context          * ctx,
                struct ggml_tensor           * a,
                       ggml_custom1_op_f32_t   fun),
            "use ggml_map_custom1 instead");
    """
    ...
  def ggml_map_custom1_inplace(ctx: ffi.CData, a: ffi.CData, fun: ffi.CData, n_tasks: int, userdata: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_map_custom1_inplace(
                struct ggml_context   * ctx,
                struct ggml_tensor    * a,
                ggml_custom1_op_t       fun,
                int                     n_tasks,
                void                  * userdata);
    """
    ...
  def ggml_map_custom1_inplace_f32(ctx: ffi.CData, a: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_custom1_inplace_f32(
                struct ggml_context          * ctx,
                struct ggml_tensor           * a,
                       ggml_custom1_op_f32_t   fun),
            "use ggml_map_custom1_inplace instead");
    """
    ...
  def ggml_map_custom2(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, fun: ffi.CData, n_tasks: int, userdata: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_map_custom2(
                struct ggml_context   * ctx,
                struct ggml_tensor    * a,
                struct ggml_tensor    * b,
                ggml_custom2_op_t       fun,
                int                     n_tasks,
                void                  * userdata);
    """
    ...
  def ggml_map_custom2_f32(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_custom2_f32(
                struct ggml_context          * ctx,
                struct ggml_tensor           * a,
                struct ggml_tensor           * b,
                       ggml_custom2_op_f32_t   fun),
            "use ggml_map_custom2 instead");
    """
    ...
  def ggml_map_custom2_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, fun: ffi.CData, n_tasks: int, userdata: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_map_custom2_inplace(
                struct ggml_context   * ctx,
                struct ggml_tensor    * a,
                struct ggml_tensor    * b,
                ggml_custom2_op_t       fun,
                int                     n_tasks,
                void                  * userdata);
    """
    ...
  def ggml_map_custom2_inplace_f32(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_custom2_inplace_f32(
                struct ggml_context          * ctx,
                struct ggml_tensor           * a,
                struct ggml_tensor           * b,
                       ggml_custom2_op_f32_t   fun),
            "use ggml_map_custom2_inplace instead");
    """
    ...
  def ggml_map_custom3(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, c: ffi.CData, fun: ffi.CData, n_tasks: int, userdata: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_map_custom3(
                struct ggml_context   * ctx,
                struct ggml_tensor    * a,
                struct ggml_tensor    * b,
                struct ggml_tensor    * c,
                ggml_custom3_op_t       fun,
                int                     n_tasks,
                void                  * userdata);
    """
    ...
  def ggml_map_custom3_f32(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, c: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_custom3_f32(
                struct ggml_context          * ctx,
                struct ggml_tensor           * a,
                struct ggml_tensor           * b,
                struct ggml_tensor           * c,
                       ggml_custom3_op_f32_t   fun),
            "use ggml_map_custom3 instead");
    """
    ...
  def ggml_map_custom3_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, c: ffi.CData, fun: ffi.CData, n_tasks: int, userdata: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_map_custom3_inplace(
                struct ggml_context   * ctx,
                struct ggml_tensor    * a,
                struct ggml_tensor    * b,
                struct ggml_tensor    * c,
                ggml_custom3_op_t       fun,
                int                     n_tasks,
                void                  * userdata);
    """
    ...
  def ggml_map_custom3_inplace_f32(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, c: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_custom3_inplace_f32(
                struct ggml_context          * ctx,
                struct ggml_tensor           * a,
                struct ggml_tensor           * b,
                struct ggml_tensor           * c,
                       ggml_custom3_op_f32_t   fun),
            "use ggml_map_custom3_inplace instead");
    """
    ...
  def ggml_map_unary_f32(ctx: ffi.CData, a: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_unary_f32(
                struct ggml_context        * ctx,
                struct ggml_tensor         * a,
                       ggml_unary_op_f32_t   fun),
            "use ggml_map_custom1 instead");
    """
    ...
  def ggml_map_unary_inplace_f32(ctx: ffi.CData, a: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_unary_inplace_f32(
                struct ggml_context        * ctx,
                struct ggml_tensor         * a,
                       ggml_unary_op_f32_t   fun),
            "use ggml_map_custom1_inplace instead");
    """
    ...
  def ggml_mean(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
    mean along rows

        GGML_API struct ggml_tensor * ggml_mean(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_mul(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_mul(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_mul_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_mul_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_mul_mat(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
    A: k columns, n rows => [ne03, ne02, n, k]
    B: k columns, m rows  (i.e. we transpose it internally) => [ne03 * x, ne02 * y, m, k]
    result is n columns, m rows => [ne03 * x, ne02 * y, m, n]

        GGML_API struct ggml_tensor * ggml_mul_mat(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_mul_mat_id(ctx: ffi.CData, as: ffi.CData, b: ffi.CData, ids: ffi.CData) -> ffi.CData:
    """
    indirect matrix multiplication

        GGML_API struct ggml_tensor * ggml_mul_mat_id(
                struct ggml_context * ctx,
                struct ggml_tensor  * as,
                struct ggml_tensor  * b,
                struct ggml_tensor  * ids);
    """
    ...
  def ggml_mul_mat_set_prec(a: ffi.CData, prec: int) -> None:
    """
    change the precision of a matrix multiplication
    set to GGML_PREC_F32 for higher precision (useful for phi-2)

        GGML_API void ggml_mul_mat_set_prec(
                struct ggml_tensor * a,
                enum ggml_prec       prec);
    """
    ...
  def ggml_n_dims(tensor: ffi.CData) -> int:
    """    GGML_API           int  ggml_n_dims       (const struct ggml_tensor * tensor); // returns 1 for scalars"""
    ...
  def ggml_nbytes(tensor: ffi.CData) -> int:
    """    GGML_API GGML_CALL size_t  ggml_nbytes      (const struct ggml_tensor * tensor);"""
    ...
  def ggml_nbytes_pad(tensor: ffi.CData) -> int:
    """    GGML_API           size_t  ggml_nbytes_pad  (const struct ggml_tensor * tensor); // same as ggml_nbytes() but padded to GGML_MEM_ALIGN"""
    ...
  def ggml_neg(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_neg(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_neg_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_neg_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_nelements(tensor: ffi.CData) -> int:
    """    GGML_API GGML_CALL int64_t ggml_nelements   (const struct ggml_tensor * tensor);"""
    ...
  def ggml_new_f32(ctx: ffi.CData, value: float) -> ffi.CData:
    """    GGML_API struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);"""
    ...
  def ggml_new_graph(ctx: ffi.CData) -> ffi.CData:
    """
    graph allocation in a context

        GGML_API struct ggml_cgraph * ggml_new_graph         (struct ggml_context * ctx); // size = GGML_DEFAULT_GRAPH_SIZE, grads = false
    """
    ...
  def ggml_new_graph_custom(ctx: ffi.CData, size: int, grads: ffi.CData) -> ffi.CData:
    """    GGML_API struct ggml_cgraph * ggml_new_graph_custom  (struct ggml_context * ctx, size_t size, bool grads);"""
    ...
  def ggml_new_i32(ctx: ffi.CData, value: int) -> ffi.CData:
    """    GGML_API struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);"""
    ...
  def ggml_new_tensor(ctx: ffi.CData, type: int, n_dims: int, ne: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_new_tensor(
                struct ggml_context * ctx,
                enum   ggml_type type,
                int    n_dims,
                const int64_t *ne);
    """
    ...
  def ggml_new_tensor_1d(ctx: ffi.CData, type: int, ne0: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_new_tensor_1d(
                struct ggml_context * ctx,
                enum   ggml_type type,
                int64_t ne0);
    """
    ...
  def ggml_new_tensor_2d(ctx: ffi.CData, type: int, ne0: int, ne1: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_new_tensor_2d(
                struct ggml_context * ctx,
                enum   ggml_type type,
                int64_t ne0,
                int64_t ne1);
    """
    ...
  def ggml_new_tensor_3d(ctx: ffi.CData, type: int, ne0: int, ne1: int, ne2: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_new_tensor_3d(
                struct ggml_context * ctx,
                enum   ggml_type type,
                int64_t ne0,
                int64_t ne1,
                int64_t ne2);
    """
    ...
  def ggml_new_tensor_4d(ctx: ffi.CData, type: int, ne0: int, ne1: int, ne2: int, ne3: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_new_tensor_4d(
                struct ggml_context * ctx,
                enum   ggml_type type,
                int64_t ne0,
                int64_t ne1,
                int64_t ne2,
                int64_t ne3);
    """
    ...
  def ggml_norm(ctx: ffi.CData, a: ffi.CData, eps: float) -> ffi.CData:
    """
    normalize along rows

        GGML_API struct ggml_tensor * ggml_norm(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                float                 eps);
    """
    ...
  def ggml_norm_inplace(ctx: ffi.CData, a: ffi.CData, eps: float) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_norm_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                float                 eps);
    """
    ...
  def ggml_nrows(tensor: ffi.CData) -> int:
    """    GGML_API GGML_CALL int64_t ggml_nrows       (const struct ggml_tensor * tensor);"""
    ...
  def ggml_numa_init(numa: int) -> None:
    """    GGML_API void    ggml_numa_init(enum ggml_numa_strategy numa); // call once for better performance on NUMA systems"""
    ...
  def ggml_op_desc(t: ffi.CData) -> ffi.CData:
    """    GGML_API GGML_CALL const char * ggml_op_desc(const struct ggml_tensor * t); // unary or op name"""
    ...
  def ggml_op_name(op: int) -> ffi.CData:
    """    GGML_API GGML_CALL const char * ggml_op_name  (enum ggml_op   op);"""
    ...
  def ggml_op_symbol(op: int) -> ffi.CData:
    """    GGML_API           const char * ggml_op_symbol(enum ggml_op   op);"""
    ...
  def ggml_opt(ctx: ffi.CData, params: ffi.CData, f: ffi.CData) -> int:
    """
    optimize the function defined by the tensor f

        GGML_API enum ggml_opt_result ggml_opt(
                struct ggml_context * ctx,
                struct ggml_opt_params params,
                struct ggml_tensor * f);
    """
    ...
  def ggml_opt_default_params(type: int) -> ffi.CData:
    """    GGML_API struct ggml_opt_params ggml_opt_default_params(enum ggml_opt_type type);"""
    ...
  def ggml_opt_init(ctx: ffi.CData, opt: ffi.CData, params: ffi.CData, nx: int) -> None:
    """
    initialize optimizer context

        GGML_API void ggml_opt_init(
                struct ggml_context     * ctx,
                struct ggml_opt_context * opt,
                struct ggml_opt_params    params,
                int64_t                   nx);
    """
    ...
  def ggml_opt_resume(ctx: ffi.CData, opt: ffi.CData, f: ffi.CData) -> int:
    """
    continue optimizing the function defined by the tensor f

        GGML_API enum ggml_opt_result ggml_opt_resume(
                struct ggml_context * ctx,
                struct ggml_opt_context * opt,
                struct ggml_tensor * f);
    """
    ...
  def ggml_opt_resume_g(ctx: ffi.CData, opt: ffi.CData, f: ffi.CData, gf: ffi.CData, gb: ffi.CData, callback: ffi.CData, callback_data: ffi.CData) -> int:
    """
    continue optimizing the function defined by the tensor f

        GGML_API enum ggml_opt_result ggml_opt_resume_g(
                struct ggml_context * ctx,
                struct ggml_opt_context * opt,
                struct ggml_tensor * f,
                struct ggml_cgraph * gf,
                struct ggml_cgraph * gb,
                ggml_opt_callback callback,
                void * callback_data);
    """
    ...
  def ggml_out_prod(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
    A: m columns, n rows,
    B: p columns, n rows,
    result is m columns, p rows

        GGML_API struct ggml_tensor * ggml_out_prod(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_pad(ctx: ffi.CData, a: ffi.CData, p0: int, p1: int, p2: int, p3: int) -> ffi.CData:
    """
    pad each dimension with zeros: [x, ..., x] -> [x, ..., x, 0, ..., 0]

        GGML_API struct ggml_tensor * ggml_pad(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                  p0,
                int                  p1,
                int                  p2,
                int                  p3);
    """
    ...
  def ggml_permute(ctx: ffi.CData, a: ffi.CData, axis0: int, axis1: int, axis2: int, axis3: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_permute(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   axis0,
                int                   axis1,
                int                   axis2,
                int                   axis3);
    """
    ...
  def ggml_pool_1d(ctx: ffi.CData, a: ffi.CData, op: int, k0: int, s0: int, p0: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_pool_1d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                enum ggml_op_pool     op,
                int                   k0, // kernel size
                int                   s0, // stride
                int                   p0); // padding
    """
    ...
  def ggml_pool_2d(ctx: ffi.CData, a: ffi.CData, op: int, k0: int, k1: int, s0: int, s1: int, p0: float, p1: float) -> ffi.CData:
    """
    the result will have 2*p0 padding for the first dimension
    and 2*p1 padding for the second dimension

        GGML_API struct ggml_tensor * ggml_pool_2d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                enum ggml_op_pool     op,
                int                   k0,
                int                   k1,
                int                   s0,
                int                   s1,
                float                 p0,
                float                 p1);
    """
    ...
  def ggml_print_backtrace() -> None:
    """    GGML_API void    ggml_print_backtrace(void);"""
    ...
  def ggml_print_object(obj: ffi.CData) -> None:
    """    GGML_API void    ggml_print_object (const struct ggml_object * obj);"""
    ...
  def ggml_print_objects(ctx: ffi.CData) -> None:
    """    GGML_API void    ggml_print_objects(const struct ggml_context * ctx);"""
    ...
  def ggml_quantize_chunk(type: int, src: ffi.CData, dst: ffi.CData, start: int, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """
    calls ggml_quantize_init internally (i.e. can allocate memory)

        GGML_API size_t ggml_quantize_chunk(
                enum ggml_type   type,
                   const float * src,
                          void * dst,
                       int64_t   start,
                       int64_t   nrows,
                       int64_t   n_per_row,
                   const float * imatrix);
    """
    ...
  def ggml_quantize_free() -> None:
    """    GGML_API void ggml_quantize_free(void);"""
    ...
  def ggml_quantize_init(type: int) -> None:
    """
    - ggml_quantize_init can be called multiple times with the same type
    it will only initialize the quantization tables for the first call or after ggml_quantize_free
    automatically called by ggml_quantize_chunk for convenience
    
    - ggml_quantize_free will free any memory allocated by ggml_quantize_init
    call this at the end of the program to avoid memory leaks
    
    note: these are thread-safe
    

        GGML_API void ggml_quantize_init(enum ggml_type type);
    """
    ...
  def ggml_quantize_requires_imatrix(type: int) -> ffi.CData:
    """
    some quantization type cannot be used without an importance matrix

        GGML_API bool ggml_quantize_requires_imatrix(enum ggml_type type);
    """
    ...
  def ggml_relu(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_relu(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_relu_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_relu_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_repeat(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
    if a is the same shape as b, and a is not parameter, return a
    otherwise, return a new tensor: repeat(a) to fit in b

        GGML_API struct ggml_tensor * ggml_repeat(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_repeat_back(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
    sums repetitions in a into shape of b

        GGML_API struct ggml_tensor * ggml_repeat_back(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_reshape(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
    return view(a), b specifies the new shape
    TODO: when we start computing gradient, make a copy instead of view

        GGML_API struct ggml_tensor * ggml_reshape(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_reshape_1d(ctx: ffi.CData, a: ffi.CData, ne0: int) -> ffi.CData:
    """
    return view(a)
    TODO: when we start computing gradient, make a copy instead of view

        GGML_API struct ggml_tensor * ggml_reshape_1d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int64_t               ne0);
    """
    ...
  def ggml_reshape_2d(ctx: ffi.CData, a: ffi.CData, ne0: int, ne1: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_reshape_2d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int64_t               ne0,
                int64_t               ne1);
    """
    ...
  def ggml_reshape_3d(ctx: ffi.CData, a: ffi.CData, ne0: int, ne1: int, ne2: int) -> ffi.CData:
    """
    return view(a)
    TODO: when we start computing gradient, make a copy instead of view

        GGML_API struct ggml_tensor * ggml_reshape_3d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int64_t               ne0,
                int64_t               ne1,
                int64_t               ne2);
    """
    ...
  def ggml_reshape_4d(ctx: ffi.CData, a: ffi.CData, ne0: int, ne1: int, ne2: int, ne3: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_reshape_4d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int64_t               ne0,
                int64_t               ne1,
                int64_t               ne2,
                int64_t               ne3);
    """
    ...
  def ggml_rms_norm(ctx: ffi.CData, a: ffi.CData, eps: float) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_rms_norm(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                float                 eps);
    """
    ...
  def ggml_rms_norm_back(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, eps: float) -> ffi.CData:
    """
    a - x
    b - dy

        GGML_API struct ggml_tensor * ggml_rms_norm_back(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                float                 eps);
    """
    ...
  def ggml_rms_norm_inplace(ctx: ffi.CData, a: ffi.CData, eps: float) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_rms_norm_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                float                 eps);
    """
    ...
  def ggml_rope(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, n_dims: int, mode: int, n_ctx: int) -> ffi.CData:
    """
    rotary position embedding
    if mode & 1 == 1, skip n_past elements (DEPRECATED)
    if mode & 2 == 1, GPT-NeoX style
    if mode & 4 == 1, ChatGLM style
    
    b is an int32 vector with size a->ne[2], it contains the positions

        GGML_API struct ggml_tensor * ggml_rope(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                int                   n_dims,
                int                   mode,
                int                   n_ctx);
    """
    ...
  def ggml_rope_back(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, n_dims: int, mode: int, n_ctx: int, n_orig_ctx: int, freq_base: float, freq_scale: float, ext_factor: float, attn_factor: float, beta_fast: float, beta_slow: float, xpos_base: float, xpos_down: ffi.CData) -> ffi.CData:
    """
    rotary position embedding backward, i.e compute dx from dy
    a - dy

        GGML_API struct ggml_tensor * ggml_rope_back(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                int                   n_dims,
                int                   mode,
                int                   n_ctx,
                int                   n_orig_ctx,
                float                 freq_base,
                float                 freq_scale,
                float                 ext_factor,
                float                 attn_factor,
                float                 beta_fast,
                float                 beta_slow,
                float                 xpos_base,
                bool                  xpos_down);
    """
    ...
  def ggml_rope_custom(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, n_dims: int, mode: int, n_ctx: int, n_orig_ctx: int, freq_base: float, freq_scale: float, ext_factor: float, attn_factor: float, beta_fast: float, beta_slow: float) -> ffi.CData:
    """
    custom RoPE

        GGML_API struct ggml_tensor * ggml_rope_custom(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                int                   n_dims,
                int                   mode,
                int                   n_ctx,
                int                   n_orig_ctx,
                float                 freq_base,
                float                 freq_scale,
                float                 ext_factor,
                float                 attn_factor,
                float                 beta_fast,
                float                 beta_slow);
    """
    ...
  def ggml_rope_custom_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, n_dims: int, mode: int, n_ctx: int, n_orig_ctx: int, freq_base: float, freq_scale: float, ext_factor: float, attn_factor: float, beta_fast: float, beta_slow: float) -> ffi.CData:
    """
    in-place, returns view(a)

        GGML_API struct ggml_tensor * ggml_rope_custom_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                int                   n_dims,
                int                   mode,
                int                   n_ctx,
                int                   n_orig_ctx,
                float                 freq_base,
                float                 freq_scale,
                float                 ext_factor,
                float                 attn_factor,
                float                 beta_fast,
                float                 beta_slow);
    """
    ...
  def ggml_rope_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, n_dims: int, mode: int, n_ctx: int) -> ffi.CData:
    """
    in-place, returns view(a)

        GGML_API struct ggml_tensor * ggml_rope_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                int                   n_dims,
                int                   mode,
                int                   n_ctx);
    """
    ...
  def ggml_rope_xpos_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, n_dims: int, base: float, down: ffi.CData) -> ffi.CData:
    """
    xPos RoPE, in-place, returns view(a)

        GGML_API struct ggml_tensor * ggml_rope_xpos_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                int                   n_dims,
                float                 base,
                bool                  down);
    """
    ...
  def ggml_rope_yarn_corr_dims(n_dims: int, n_orig_ctx: int, freq_base: float, beta_fast: float, beta_slow: float, dims: ffi.CData) -> None:
    """
    compute correction dims for YaRN RoPE scaling

        GGML_CALL void ggml_rope_yarn_corr_dims(
            int n_dims, int n_orig_ctx, float freq_base, float beta_fast, float beta_slow, float dims[2]);
    """
    ...
  def ggml_row_size(type: int, ne: int) -> int:
    """    GGML_API GGML_CALL size_t ggml_row_size (enum ggml_type type, int64_t ne); // size in bytes for all elements in a row"""
    ...
  def ggml_scale(ctx: ffi.CData, a: ffi.CData, s: float) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_scale(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                float                 s);
    """
    ...
  def ggml_scale_inplace(ctx: ffi.CData, a: ffi.CData, s: float) -> ffi.CData:
    """
    in-place, returns view(a)

        GGML_API struct ggml_tensor * ggml_scale_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                float                 s);
    """
    ...
  def ggml_set(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, nb1: int, nb2: int, nb3: int, offset: int) -> ffi.CData:
    """
    b -> view(a,offset,nb1,nb2,3), return modified a

        GGML_API struct ggml_tensor * ggml_set(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                size_t                nb1,
                size_t                nb2,
                size_t                nb3,
                size_t                offset);
    """
    ...
  def ggml_set_1d(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, offset: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_set_1d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                size_t                offset);
    """
    ...
  def ggml_set_1d_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, offset: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_set_1d_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                size_t                offset);
    """
    ...
  def ggml_set_2d(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, nb1: int, offset: int) -> ffi.CData:
    """
    b -> view(a,offset,nb1,nb2,3), return modified a

        GGML_API struct ggml_tensor * ggml_set_2d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                size_t                nb1,
                size_t                offset);
    """
    ...
  def ggml_set_2d_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, nb1: int, offset: int) -> ffi.CData:
    """
    b -> view(a,offset,nb1,nb2,3), return view(a)

        GGML_API struct ggml_tensor * ggml_set_2d_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                size_t                nb1,
                size_t                offset);
    """
    ...
  def ggml_set_f32(tensor: ffi.CData, value: float) -> ffi.CData:
    """    GGML_API struct ggml_tensor * ggml_set_f32 (struct ggml_tensor * tensor, float value);"""
    ...
  def ggml_set_f32_1d(tensor: ffi.CData, i: int, value: float) -> None:
    """    GGML_API void    ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);"""
    ...
  def ggml_set_f32_nd(tensor: ffi.CData, i0: int, i1: int, i2: int, i3: int, value: float) -> None:
    """    GGML_API void    ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);"""
    ...
  def ggml_set_i32(tensor: ffi.CData, value: int) -> ffi.CData:
    """    GGML_API struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);"""
    ...
  def ggml_set_i32_1d(tensor: ffi.CData, i: int, value: int) -> None:
    """    GGML_API void    ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);"""
    ...
  def ggml_set_i32_nd(tensor: ffi.CData, i0: int, i1: int, i2: int, i3: int, value: int) -> None:
    """    GGML_API void    ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);"""
    ...
  def ggml_set_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, nb1: int, nb2: int, nb3: int, offset: int) -> ffi.CData:
    """
    b -> view(a,offset,nb1,nb2,3), return view(a)

        GGML_API struct ggml_tensor * ggml_set_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b,
                size_t                nb1,
                size_t                nb2,
                size_t                nb3,
                size_t                offset);
    """
    ...
  def ggml_set_input(tensor: ffi.CData) -> None:
    """
    
    tensor flags
    

        GGML_API void ggml_set_input(struct ggml_tensor * tensor);
    """
    ...
  def ggml_set_name(tensor: ffi.CData, name: ffi.CData) -> ffi.CData:
    """    GGML_API struct ggml_tensor * ggml_set_name   (      struct ggml_tensor * tensor, const char * name);"""
    ...
  def ggml_set_no_alloc(ctx: ffi.CData, no_alloc: ffi.CData) -> None:
    """    GGML_API void    ggml_set_no_alloc(struct ggml_context * ctx, bool no_alloc);"""
    ...
  def ggml_set_output(tensor: ffi.CData) -> None:
    """    GGML_API void ggml_set_output(struct ggml_tensor * tensor);"""
    ...
  def ggml_set_param(ctx: ffi.CData, tensor: ffi.CData) -> None:
    """
        GGML_API void ggml_set_param(
                struct ggml_context * ctx,
                struct ggml_tensor  * tensor);
    """
    ...
  def ggml_set_scratch(ctx: ffi.CData, scratch: ffi.CData) -> int:
    """    GGML_API size_t  ggml_set_scratch (struct ggml_context * ctx, struct ggml_scratch scratch);"""
    ...
  def ggml_set_zero(tensor: ffi.CData) -> ffi.CData:
    """    GGML_API struct ggml_tensor * ggml_set_zero(struct ggml_tensor * tensor);"""
    ...
  def ggml_sgn(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_sgn(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_sgn_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_sgn_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_sigmoid(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_sigmoid(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_sigmoid_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_sigmoid_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_silu(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_silu(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_silu_back(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
    a - x
    b - dy

        GGML_API struct ggml_tensor * ggml_silu_back(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_silu_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_silu_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_soft_max(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_soft_max(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_soft_max_back(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_soft_max_back(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_soft_max_back_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
    in-place, returns view(a)

        GGML_API struct ggml_tensor * ggml_soft_max_back_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_soft_max_ext(ctx: ffi.CData, a: ffi.CData, mask: ffi.CData, scale: float, max_bias: float) -> ffi.CData:
    """
    fused soft_max(a*scale + mask*(ALiBi slope))
    mask is optional
    max_bias = 0.0f for no ALiBi

        GGML_API struct ggml_tensor * ggml_soft_max_ext(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * mask,
                float                 scale,
                float                 max_bias);
    """
    ...
  def ggml_soft_max_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
    in-place, returns view(a)

        GGML_API struct ggml_tensor * ggml_soft_max_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_sqr(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_sqr(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_sqr_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_sqr_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_sqrt(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_sqrt(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_sqrt_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_sqrt_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_ssm_conv(ctx: ffi.CData, s: ffi.CData, x: ffi.CData, c: ffi.CData, sq: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_ssm_conv(
                struct ggml_context * ctx,
                struct ggml_tensor  * s,
                struct ggml_tensor  * x,
                struct ggml_tensor  * c,
                struct ggml_tensor  * sq);
    """
    ...
  def ggml_ssm_scan(ctx: ffi.CData, s: ffi.CData, x: ffi.CData, dt: ffi.CData, A: ffi.CData, B: ffi.CData, C: ffi.CData, sq: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_ssm_scan(
                struct ggml_context * ctx,
                struct ggml_tensor  * s,
                struct ggml_tensor  * x,
                struct ggml_tensor  * dt,
                struct ggml_tensor  * A,
                struct ggml_tensor  * B,
                struct ggml_tensor  * C,
                struct ggml_tensor  * sq);
    """
    ...
  def ggml_status_to_string(status: int) -> ffi.CData:
    """
    get ggml_status name string

        GGML_API GGML_CALL const char * ggml_status_to_string(enum ggml_status status);
    """
    ...
  def ggml_step(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_step(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_step_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_step_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_sub(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_sub(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_sub_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_sub_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                struct ggml_tensor  * b);
    """
    ...
  def ggml_sum(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
    return scalar

        GGML_API struct ggml_tensor * ggml_sum(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_sum_rows(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
    sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]

        GGML_API struct ggml_tensor * ggml_sum_rows(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_sycl_get_device_description(device: int, description: ffi.CData, description_size: int) -> None:
    """GGML_API GGML_CALL void   ggml_sycl_get_device_description(int device, char *description, size_t description_size);"""
    ...
  def ggml_sycl_get_gpu_list(id_list: ffi.CData, max_len: int) -> None:
    """GGML_API GGML_CALL void   ggml_sycl_get_gpu_list(int *id_list, int max_len);"""
    ...
  def ggml_tallocr_alloc(talloc: ffi.CData, tensor: ffi.CData) -> None:
    """GGML_API void                ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor);"""
    ...
  def ggml_tallocr_new(buffer: ffi.CData) -> ffi.CData:
    """GGML_API struct ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer);"""
    ...
  def ggml_tanh(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_tanh(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_tanh_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_tanh_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_tensor_overhead() -> int:
    """
    use this to compute the memory overhead of a tensor

        GGML_API size_t ggml_tensor_overhead(void);
    """
    ...
  def ggml_time_init() -> None:
    """    GGML_API void    ggml_time_init(void); // call this once at the beginning of the program"""
    ...
  def ggml_time_ms() -> int:
    """    GGML_API int64_t ggml_time_ms(void);"""
    ...
  def ggml_time_us() -> int:
    """    GGML_API int64_t ggml_time_us(void);"""
    ...
  def ggml_timestep_embedding(ctx: ffi.CData, timesteps: ffi.CData, dim: int, max_period: int) -> ffi.CData:
    """
    Ref: https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/util.py#L151
    timesteps: [N,]
    return: [N, dim]

        GGML_API struct ggml_tensor * ggml_timestep_embedding(
                struct ggml_context * ctx,
                struct ggml_tensor  * timesteps,
                int                   dim,
                int                   max_period);
    """
    ...
  def ggml_top_k(ctx: ffi.CData, a: ffi.CData, k: int) -> ffi.CData:
    """
    top k elements per row

        GGML_API struct ggml_tensor * ggml_top_k(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   k);
    """
    ...
  def ggml_transpose(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
    alias for ggml_permute(ctx, a, 1, 0, 2, 3)

        GGML_API struct ggml_tensor * ggml_transpose(
                struct ggml_context * ctx,
                struct ggml_tensor  * a);
    """
    ...
  def ggml_type_name(type: int) -> ffi.CData:
    """    GGML_API GGML_CALL const char * ggml_type_name(enum ggml_type type);"""
    ...
  def ggml_type_size(type: int) -> int:
    """    GGML_API GGML_CALL size_t ggml_type_size(enum ggml_type type);             // size in bytes for all elements in a block"""
    ...
  def ggml_type_sizef(type: int) -> float:
    """
        GGML_DEPRECATED(
        GGML_API double ggml_type_sizef(enum ggml_type type), // ggml_type_size()/ggml_blck_size() as float
        "use ggml_row_size() instead");
    """
    ...
  def ggml_unary(ctx: ffi.CData, a: ffi.CData, op: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_unary(
                struct ggml_context * ctx,
                 struct ggml_tensor * a,
                 enum ggml_unary_op op);
    """
    ...
  def ggml_unary_inplace(ctx: ffi.CData, a: ffi.CData, op: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_unary_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            enum ggml_unary_op op);
    """
    ...
  def ggml_unary_op_name(op: int) -> ffi.CData:
    """    GGML_API           const char * ggml_unary_op_name(enum ggml_unary_op op);"""
    ...
  def ggml_unravel_index(tensor: ffi.CData, i: int, i0: ffi.CData, i1: ffi.CData, i2: ffi.CData, i3: ffi.CData) -> None:
    """
    Converts a flat index into coordinates

        GGML_API void    ggml_unravel_index(const struct ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3);
    """
    ...
  def ggml_upscale(ctx: ffi.CData, a: ffi.CData, scale_factor: int) -> ffi.CData:
    """
    nearest interpolate
    multiplies ne0 and ne1 by scale factor
    used in stable-diffusion

        GGML_API struct ggml_tensor * ggml_upscale(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   scale_factor);
    """
    ...
  def ggml_upscale_ext(ctx: ffi.CData, a: ffi.CData, ne0: int, ne1: int, ne2: int, ne3: int) -> ffi.CData:
    """
    nearest interpolate
    nearest interpolate to specified dimensions
    used in tortoise.cpp

        GGML_API struct ggml_tensor * ggml_upscale_ext(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   ne0,
                int                   ne1,
                int                   ne2,
                int                   ne3);
    """
    ...
  def ggml_used_mem(ctx: ffi.CData) -> int:
    """    GGML_API size_t  ggml_used_mem(const struct ggml_context * ctx);"""
    ...
  def ggml_validate_row_data(type: int, data: ffi.CData, nbytes: int) -> ffi.CData:
    """    GGML_API bool ggml_validate_row_data(enum ggml_type type, const void * data, size_t nbytes);"""
    ...
  def ggml_vec_dot_iq1_m_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_iq1_m_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_iq1_s_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_iq1_s_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_iq2_s_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_iq2_s_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_iq2_xs_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_iq2_xs_q8_K (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_iq2_xxs_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_iq2_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_iq3_s_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_iq3_s_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_iq3_xxs_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_iq3_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_iq4_nl_q8_0(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_iq4_nl_q8_0 (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_iq4_xs_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_iq4_xs_q8_K (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_q2_K_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_q2_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_q3_K_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_q3_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_q4_0_q8_0(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """
    Dot product

    void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
    """
    ...
  def ggml_vec_dot_q4_1_q8_1(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_q4_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_q4_K_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_q4_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_q5_0_q8_0(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_q5_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_q5_1_q8_1(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_q5_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_q5_K_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_q5_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_q6_K_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_q8_0_q8_0(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_q8_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);"""
    ...
  def ggml_view_1d(ctx: ffi.CData, a: ffi.CData, ne0: int, offset: int) -> ffi.CData:
    """
    offset in bytes

        GGML_API struct ggml_tensor * ggml_view_1d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int64_t               ne0,
                size_t                offset);
    """
    ...
  def ggml_view_2d(ctx: ffi.CData, a: ffi.CData, ne0: int, ne1: int, nb1: int, offset: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_view_2d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int64_t               ne0,
                int64_t               ne1,
                size_t                nb1, // row stride in bytes
                size_t                offset);
    """
    ...
  def ggml_view_3d(ctx: ffi.CData, a: ffi.CData, ne0: int, ne1: int, ne2: int, nb1: int, nb2: int, offset: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_view_3d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int64_t               ne0,
                int64_t               ne1,
                int64_t               ne2,
                size_t                nb1, // row   stride in bytes
                size_t                nb2, // slice stride in bytes
                size_t                offset);
    """
    ...
  def ggml_view_4d(ctx: ffi.CData, a: ffi.CData, ne0: int, ne1: int, ne2: int, ne3: int, nb1: int, nb2: int, nb3: int, offset: int) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_view_4d(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int64_t               ne0,
                int64_t               ne1,
                int64_t               ne2,
                int64_t               ne3,
                size_t                nb1, // row   stride in bytes
                size_t                nb2, // slice stride in bytes
                size_t                nb3,
                size_t                offset);
    """
    ...
  def ggml_view_tensor(ctx: ffi.CData, src: ffi.CData) -> ffi.CData:
    """    GGML_API struct ggml_tensor * ggml_view_tensor(struct ggml_context * ctx, struct ggml_tensor * src);"""
    ...
  def ggml_vk_available_devices(memoryRequired: int, count: ffi.CData) -> ffi.CData:
    """struct ggml_vk_device * ggml_vk_available_devices(size_t memoryRequired, size_t * count);"""
    ...
  def ggml_vk_current_device() -> ffi.CData:
    """struct ggml_vk_device ggml_vk_current_device(void);"""
    ...
  def ggml_vk_get_device(device: ffi.CData, memoryRequired: int, name: ffi.CData) -> ffi.CData:
    """bool ggml_vk_get_device(struct ggml_vk_device * device, size_t memoryRequired, const char * name);"""
    ...
  def ggml_vk_has_device() -> ffi.CData:
    """bool ggml_vk_has_device(void);"""
    ...
  def ggml_vk_has_vulkan() -> ffi.CData:
    """bool ggml_vk_has_vulkan(void);"""
    ...
  def ggml_vk_instance_init() -> None:
    """GGML_API void ggml_vk_instance_init(void);"""
    ...
  def ggml_win_part(ctx: ffi.CData, a: ffi.CData, w: int) -> ffi.CData:
    """
    partition into non-overlapping windows with padding if needed
    example:
    a:   768   64   64    1
    w:    14
    res: 768   14   14    25
    used in sam

        GGML_API struct ggml_tensor * ggml_win_part(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   w);
    """
    ...
  def ggml_win_unpart(ctx: ffi.CData, a: ffi.CData, w0: int, h0: int, w: int) -> ffi.CData:
    """
    reverse of ggml_win_part
    used in sam

        GGML_API struct ggml_tensor * ggml_win_unpart(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                int                   w0,
                int                   h0,
                int                   w);
    """
    ...
  def gguf_add_tensor(ctx: ffi.CData, tensor: ffi.CData) -> None:
    """
    manage tensor info

        GGML_API void gguf_add_tensor(struct gguf_context * ctx, const struct ggml_tensor * tensor);
    """
    ...
  def gguf_find_key(ctx: ffi.CData, key: ffi.CData) -> int:
    """    GGML_API int          gguf_find_key(const struct gguf_context * ctx, const char * key);"""
    ...
  def gguf_find_tensor(ctx: ffi.CData, name: ffi.CData) -> int:
    """    GGML_API int            gguf_find_tensor      (const struct gguf_context * ctx, const char * name);"""
    ...
  def gguf_free(ctx: ffi.CData) -> None:
    """    GGML_API void gguf_free(struct gguf_context * ctx);"""
    ...
  def gguf_get_alignment(ctx: ffi.CData) -> int:
    """    GGML_API size_t gguf_get_alignment  (const struct gguf_context * ctx);"""
    ...
  def gguf_get_arr_data(ctx: ffi.CData, key_id: int) -> ffi.CData:
    """    GGML_API const void * gguf_get_arr_data(const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_arr_n(ctx: ffi.CData, key_id: int) -> int:
    """    GGML_API int          gguf_get_arr_n   (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_arr_str(ctx: ffi.CData, key_id: int, i: int) -> ffi.CData:
    """    GGML_API const char * gguf_get_arr_str (const struct gguf_context * ctx, int key_id, int i);"""
    ...
  def gguf_get_arr_type(ctx: ffi.CData, key_id: int) -> int:
    """    GGML_API enum gguf_type gguf_get_arr_type(const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_data(ctx: ffi.CData) -> ffi.CData:
    """    GGML_API void * gguf_get_data       (const struct gguf_context * ctx);"""
    ...
  def gguf_get_data_offset(ctx: ffi.CData) -> int:
    """    GGML_API size_t gguf_get_data_offset(const struct gguf_context * ctx);"""
    ...
  def gguf_get_key(ctx: ffi.CData, key_id: int) -> ffi.CData:
    """    GGML_API const char * gguf_get_key (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_kv_type(ctx: ffi.CData, key_id: int) -> int:
    """    GGML_API enum gguf_type gguf_get_kv_type (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_meta_data(ctx: ffi.CData, data: ffi.CData) -> None:
    """    GGML_API void   gguf_get_meta_data(const struct gguf_context * ctx, void * data);"""
    ...
  def gguf_get_meta_size(ctx: ffi.CData) -> int:
    """
    get the size in bytes of the meta data (header, kv pairs, tensor info) including padding

        GGML_API size_t gguf_get_meta_size(const struct gguf_context * ctx);
    """
    ...
  def gguf_get_n_kv(ctx: ffi.CData) -> int:
    """    GGML_API int          gguf_get_n_kv(const struct gguf_context * ctx);"""
    ...
  def gguf_get_n_tensors(ctx: ffi.CData) -> int:
    """    GGML_API int            gguf_get_n_tensors    (const struct gguf_context * ctx);"""
    ...
  def gguf_get_tensor_name(ctx: ffi.CData, i: int) -> ffi.CData:
    """    GGML_API char *         gguf_get_tensor_name  (const struct gguf_context * ctx, int i);"""
    ...
  def gguf_get_tensor_nbytes(ctx: ffi.CData, name: ffi.CData) -> int:
    """    GGML_API size_t         gguf_get_tensor_nbytes(const struct gguf_context * ctx, const char * name);"""
    ...
  def gguf_get_tensor_offset(ctx: ffi.CData, i: int) -> int:
    """    GGML_API size_t         gguf_get_tensor_offset(const struct gguf_context * ctx, int i);"""
    ...
  def gguf_get_tensor_type(ctx: ffi.CData, i: int) -> int:
    """    GGML_API enum ggml_type gguf_get_tensor_type  (const struct gguf_context * ctx, int i);"""
    ...
  def gguf_get_val_bool(ctx: ffi.CData, key_id: int) -> ffi.CData:
    """    GGML_API bool         gguf_get_val_bool(const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_val_data(ctx: ffi.CData, key_id: int) -> ffi.CData:
    """    GGML_API const void * gguf_get_val_data(const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_val_f32(ctx: ffi.CData, key_id: int) -> float:
    """    GGML_API float        gguf_get_val_f32 (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_val_f64(ctx: ffi.CData, key_id: int) -> float:
    """    GGML_API double       gguf_get_val_f64 (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_val_i16(ctx: ffi.CData, key_id: int) -> int:
    """    GGML_API int16_t      gguf_get_val_i16 (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_val_i32(ctx: ffi.CData, key_id: int) -> int:
    """    GGML_API int32_t      gguf_get_val_i32 (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_val_i64(ctx: ffi.CData, key_id: int) -> int:
    """    GGML_API int64_t      gguf_get_val_i64 (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_val_i8(ctx: ffi.CData, key_id: int) -> int:
    """    GGML_API int8_t       gguf_get_val_i8  (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_val_str(ctx: ffi.CData, key_id: int) -> ffi.CData:
    """    GGML_API const char * gguf_get_val_str (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_val_u16(ctx: ffi.CData, key_id: int) -> int:
    """    GGML_API uint16_t     gguf_get_val_u16 (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_val_u32(ctx: ffi.CData, key_id: int) -> int:
    """    GGML_API uint32_t     gguf_get_val_u32 (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_val_u64(ctx: ffi.CData, key_id: int) -> int:
    """    GGML_API uint64_t     gguf_get_val_u64 (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_val_u8(ctx: ffi.CData, key_id: int) -> int:
    """
    will abort if the wrong type is used for the key

        GGML_API uint8_t      gguf_get_val_u8  (const struct gguf_context * ctx, int key_id);
    """
    ...
  def gguf_get_version(ctx: ffi.CData) -> int:
    """    GGML_API int    gguf_get_version    (const struct gguf_context * ctx);"""
    ...
  def gguf_init_empty() -> ffi.CData:
    """    GGML_API struct gguf_context * gguf_init_empty(void);"""
    ...
  def gguf_init_from_file(fname: ffi.CData, params: ffi.CData) -> ffi.CData:
    """    GGML_API struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params);"""
    ...
  def gguf_remove_key(ctx: ffi.CData, key: ffi.CData) -> None:
    """
    removes key if it exists

        GGML_API void gguf_remove_key(struct gguf_context * ctx, const char * key);
    """
    ...
  def gguf_set_arr_data(ctx: ffi.CData, key: ffi.CData, type: int, data: ffi.CData, n: int) -> None:
    """    GGML_API void gguf_set_arr_data(struct gguf_context * ctx, const char * key, enum gguf_type type, const void * data, int n);"""
    ...
  def gguf_set_arr_str(ctx: ffi.CData, key: ffi.CData, data: ffi.CData, n: int) -> None:
    """    GGML_API void gguf_set_arr_str (struct gguf_context * ctx, const char * key, const char ** data, int n);"""
    ...
  def gguf_set_kv(ctx: ffi.CData, src: ffi.CData) -> None:
    """
    set or add KV pairs from another context

        GGML_API void gguf_set_kv(struct gguf_context * ctx, struct gguf_context * src);
    """
    ...
  def gguf_set_tensor_data(ctx: ffi.CData, name: ffi.CData, data: ffi.CData, size: int) -> None:
    """    GGML_API void gguf_set_tensor_data(struct gguf_context * ctx, const char * name, const void * data, size_t size);"""
    ...
  def gguf_set_tensor_type(ctx: ffi.CData, name: ffi.CData, type: int) -> None:
    """    GGML_API void gguf_set_tensor_type(struct gguf_context * ctx, const char * name, enum ggml_type type);"""
    ...
  def gguf_set_val_bool(ctx: ffi.CData, key: ffi.CData, val: ffi.CData) -> None:
    """    GGML_API void gguf_set_val_bool(struct gguf_context * ctx, const char * key, bool     val);"""
    ...
  def gguf_set_val_f32(ctx: ffi.CData, key: ffi.CData, val: float) -> None:
    """    GGML_API void gguf_set_val_f32 (struct gguf_context * ctx, const char * key, float    val);"""
    ...
  def gguf_set_val_f64(ctx: ffi.CData, key: ffi.CData, val: float) -> None:
    """    GGML_API void gguf_set_val_f64 (struct gguf_context * ctx, const char * key, double   val);"""
    ...
  def gguf_set_val_i16(ctx: ffi.CData, key: ffi.CData, val: int) -> None:
    """    GGML_API void gguf_set_val_i16 (struct gguf_context * ctx, const char * key, int16_t  val);"""
    ...
  def gguf_set_val_i32(ctx: ffi.CData, key: ffi.CData, val: int) -> None:
    """    GGML_API void gguf_set_val_i32 (struct gguf_context * ctx, const char * key, int32_t  val);"""
    ...
  def gguf_set_val_i64(ctx: ffi.CData, key: ffi.CData, val: int) -> None:
    """    GGML_API void gguf_set_val_i64 (struct gguf_context * ctx, const char * key, int64_t  val);"""
    ...
  def gguf_set_val_i8(ctx: ffi.CData, key: ffi.CData, val: int) -> None:
    """    GGML_API void gguf_set_val_i8  (struct gguf_context * ctx, const char * key, int8_t   val);"""
    ...
  def gguf_set_val_str(ctx: ffi.CData, key: ffi.CData, val: ffi.CData) -> None:
    """    GGML_API void gguf_set_val_str (struct gguf_context * ctx, const char * key, const char * val);"""
    ...
  def gguf_set_val_u16(ctx: ffi.CData, key: ffi.CData, val: int) -> None:
    """    GGML_API void gguf_set_val_u16 (struct gguf_context * ctx, const char * key, uint16_t val);"""
    ...
  def gguf_set_val_u32(ctx: ffi.CData, key: ffi.CData, val: int) -> None:
    """    GGML_API void gguf_set_val_u32 (struct gguf_context * ctx, const char * key, uint32_t val);"""
    ...
  def gguf_set_val_u64(ctx: ffi.CData, key: ffi.CData, val: int) -> None:
    """    GGML_API void gguf_set_val_u64 (struct gguf_context * ctx, const char * key, uint64_t val);"""
    ...
  def gguf_set_val_u8(ctx: ffi.CData, key: ffi.CData, val: int) -> None:
    """
    overrides existing values or adds a new one

        GGML_API void gguf_set_val_u8  (struct gguf_context * ctx, const char * key, uint8_t  val);
    """
    ...
  def gguf_type_name(type: int) -> ffi.CData:
    """    GGML_API const char * gguf_type_name(enum gguf_type type);"""
    ...
  def gguf_write_to_file(ctx: ffi.CData, fname: ffi.CData, only_meta: ffi.CData) -> None:
    """
    write the entire context to a binary file

        GGML_API void gguf_write_to_file(const struct gguf_context * ctx, const char * fname, bool only_meta);
    """
    ...
  def iq2xs_free_impl(type: int) -> None:
    """void iq2xs_free_impl(enum ggml_type type);"""
    ...
  def iq2xs_init_impl(type: int) -> None:
    """void iq2xs_init_impl(enum ggml_type type);"""
    ...
  def iq3xs_free_impl(grid_size: int) -> None:
    """void iq3xs_free_impl(int grid_size);"""
    ...
  def iq3xs_init_impl(grid_size: int) -> None:
    """void iq3xs_init_impl(int grid_size);"""
    ...
  def quantize_iq1_m(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_iq1_m  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);"""
    ...
  def quantize_iq1_s(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_iq1_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);"""
    ...
  def quantize_iq2_s(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_iq2_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);"""
    ...
  def quantize_iq2_xs(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_iq2_xs (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);"""
    ...
  def quantize_iq2_xxs(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """
    Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")

    size_t quantize_iq2_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
    """
    ...
  def quantize_iq3_s(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_iq3_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);"""
    ...
  def quantize_iq3_xxs(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_iq3_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);"""
    ...
  def quantize_iq4_nl(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_iq4_nl (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);"""
    ...
  def quantize_iq4_xs(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_iq4_xs (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);"""
    ...
  def quantize_q2_K(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_q2_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);"""
    ...
  def quantize_q3_K(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_q3_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);"""
    ...
  def quantize_q4_0(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_q4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);"""
    ...
  def quantize_q4_1(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_q4_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);"""
    ...
  def quantize_q4_K(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_q4_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);"""
    ...
  def quantize_q5_0(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_q5_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);"""
    ...
  def quantize_q5_1(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_q5_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);"""
    ...
  def quantize_q5_K(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_q5_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);"""
    ...
  def quantize_q6_K(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_q6_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);"""
    ...
  def quantize_q8_0(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_q8_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);"""
    ...
  def quantize_row_iq2_s(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_iq2_s  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_iq2_s_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_iq2_s_reference  (const float * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_iq3_s(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_iq3_s  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_iq3_s_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_iq3_s_reference  (const float * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_iq3_xxs(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_iq3_xxs(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_iq3_xxs_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_iq3_xxs_reference(const float * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_iq4_nl(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_iq4_nl (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_iq4_nl_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_iq4_nl_reference (const float * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_iq4_xs(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_iq4_xs (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_iq4_xs_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_iq4_xs_reference (const float * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_q2_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q2_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_q2_K_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q2_K_reference(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_q3_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q3_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_q3_K_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q3_K_reference(const float * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_q4_0(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_q4_0_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """
    Quantization

    void quantize_row_q4_0_reference(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);
    """
    ...
  def quantize_row_q4_1(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q4_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_q4_1_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q4_1_reference(const float * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_q4_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q4_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_q4_K_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q4_K_reference(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_q5_0(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q5_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_q5_0_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q5_0_reference(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_q5_1(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q5_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_q5_1_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q5_1_reference(const float * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_q5_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q5_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_q5_K_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q5_K_reference(const float * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_q6_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q6_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_q6_K_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q6_K_reference(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_q8_0(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q8_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_q8_0_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q8_0_reference(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_q8_1(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q8_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_q8_1_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q8_1_reference(const float * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_q8_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q8_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);"""
    ...
  def quantize_row_q8_K_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q8_K_reference(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);"""
    ...
  def start_rpc_server(backend: ffi.CData, endpoint: ffi.CData, free_mem: int, total_mem: int) -> None:
    """GGML_API GGML_CALL void start_rpc_server(ggml_backend_t backend, const char * endpoint, size_t free_mem, size_t total_mem);"""
    ...