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
  @property
  def LLAMA_FTYPE_ALL_F32(self) -> int: ...
  @property
  def LLAMA_FTYPE_GUESSED(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_BF16(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_F16(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_IQ1_M(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_IQ1_S(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_IQ2_M(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_IQ2_S(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_IQ2_XS(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_IQ2_XXS(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_IQ3_M(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_IQ3_S(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_IQ3_XS(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_IQ3_XXS(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_IQ4_NL(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_IQ4_XS(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_Q2_K(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_Q2_K_S(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_Q3_K_L(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_Q3_K_M(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_Q3_K_S(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_Q4_0(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_Q4_1(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_Q4_K_M(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_Q4_K_S(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_Q5_0(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_Q5_1(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_Q5_K_M(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_Q5_K_S(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_Q6_K(self) -> int: ...
  @property
  def LLAMA_FTYPE_MOSTLY_Q8_0(self) -> int: ...
  @property
  def LLAMA_GRETYPE_ALT(self) -> int: ...
  @property
  def LLAMA_GRETYPE_CHAR(self) -> int: ...
  @property
  def LLAMA_GRETYPE_CHAR_ALT(self) -> int: ...
  @property
  def LLAMA_GRETYPE_CHAR_NOT(self) -> int: ...
  @property
  def LLAMA_GRETYPE_CHAR_RNG_UPPER(self) -> int: ...
  @property
  def LLAMA_GRETYPE_END(self) -> int: ...
  @property
  def LLAMA_GRETYPE_RULE_REF(self) -> int: ...
  @property
  def LLAMA_KV_OVERRIDE_TYPE_BOOL(self) -> int: ...
  @property
  def LLAMA_KV_OVERRIDE_TYPE_FLOAT(self) -> int: ...
  @property
  def LLAMA_KV_OVERRIDE_TYPE_INT(self) -> int: ...
  @property
  def LLAMA_KV_OVERRIDE_TYPE_STR(self) -> int: ...
  @property
  def LLAMA_POOLING_TYPE_CLS(self) -> int: ...
  @property
  def LLAMA_POOLING_TYPE_MEAN(self) -> int: ...
  @property
  def LLAMA_POOLING_TYPE_NONE(self) -> int: ...
  @property
  def LLAMA_POOLING_TYPE_UNSPECIFIED(self) -> int: ...
  @property
  def LLAMA_ROPE_SCALING_TYPE_LINEAR(self) -> int: ...
  @property
  def LLAMA_ROPE_SCALING_TYPE_MAX_VALUE(self) -> int: ...
  @property
  def LLAMA_ROPE_SCALING_TYPE_NONE(self) -> int: ...
  @property
  def LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED(self) -> int: ...
  @property
  def LLAMA_ROPE_SCALING_TYPE_YARN(self) -> int: ...
  @property
  def LLAMA_ROPE_TYPE_GLM(self) -> int: ...
  @property
  def LLAMA_ROPE_TYPE_NEOX(self) -> int: ...
  @property
  def LLAMA_ROPE_TYPE_NONE(self) -> int: ...
  @property
  def LLAMA_ROPE_TYPE_NORM(self) -> int: ...
  @property
  def LLAMA_SPLIT_MODE_LAYER(self) -> int: ...
  @property
  def LLAMA_SPLIT_MODE_NONE(self) -> int: ...
  @property
  def LLAMA_SPLIT_MODE_ROW(self) -> int: ...
  @property
  def LLAMA_TOKEN_TYPE_BYTE(self) -> int: ...
  @property
  def LLAMA_TOKEN_TYPE_CONTROL(self) -> int: ...
  @property
  def LLAMA_TOKEN_TYPE_NORMAL(self) -> int: ...
  @property
  def LLAMA_TOKEN_TYPE_UNDEFINED(self) -> int: ...
  @property
  def LLAMA_TOKEN_TYPE_UNKNOWN(self) -> int: ...
  @property
  def LLAMA_TOKEN_TYPE_UNUSED(self) -> int: ...
  @property
  def LLAMA_TOKEN_TYPE_USER_DEFINED(self) -> int: ...
  @property
  def LLAMA_VOCAB_PRE_TYPE_COMMAND_R(self) -> int: ...
  @property
  def LLAMA_VOCAB_PRE_TYPE_DBRX(self) -> int: ...
  @property
  def LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER(self) -> int: ...
  @property
  def LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM(self) -> int: ...
  @property
  def LLAMA_VOCAB_PRE_TYPE_DEFAULT(self) -> int: ...
  @property
  def LLAMA_VOCAB_PRE_TYPE_FALCON(self) -> int: ...
  @property
  def LLAMA_VOCAB_PRE_TYPE_GPT2(self) -> int: ...
  @property
  def LLAMA_VOCAB_PRE_TYPE_LLAMA3(self) -> int: ...
  @property
  def LLAMA_VOCAB_PRE_TYPE_MPT(self) -> int: ...
  @property
  def LLAMA_VOCAB_PRE_TYPE_OLMO(self) -> int: ...
  @property
  def LLAMA_VOCAB_PRE_TYPE_QWEN2(self) -> int: ...
  @property
  def LLAMA_VOCAB_PRE_TYPE_REFACT(self) -> int: ...
  @property
  def LLAMA_VOCAB_PRE_TYPE_STARCODER(self) -> int: ...
  @property
  def LLAMA_VOCAB_TYPE_BPE(self) -> int: ...
  @property
  def LLAMA_VOCAB_TYPE_NONE(self) -> int: ...
  @property
  def LLAMA_VOCAB_TYPE_SPM(self) -> int: ...
  @property
  def LLAMA_VOCAB_TYPE_WPM(self) -> int: ...
  def asprintf(arg: ffi.CData, arg2: ffi.CData, *args2) -> int:
    """int	 asprintf(char ** __restrict, const char * __restrict, ...) __printflike(2, 3);"""
    ...
  def clearerr(arg: ffi.CData) -> None:
    """void	 clearerr(FILE *);"""
    ...
  def ctermid(arg: ffi.CData) -> ffi.CData:
    """char    *ctermid(char *);"""
    ...
  def ctermid_r(arg: ffi.CData) -> ffi.CData:
    """char	*ctermid_r(char *);"""
    ...
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
  def dprintf(arg: int, arg2: ffi.CData, *args2) -> int:
    """int	dprintf(int, const char * __restrict, ...) __printflike(2, 3) __OSX_AVAILABLE_STARTING(__MAC_10_7, __IPHONE_4_3);"""
    ...
  def fclose(arg: ffi.CData) -> int:
    """int	 fclose(FILE *);"""
    ...
  def fdopen(arg: int, arg2: ffi.CData) -> ffi.CData:
    """FILE	*fdopen(int, const char *) __DARWIN_ALIAS_STARTING(__MAC_10_6, __IPHONE_2_0, __DARWIN_ALIAS(fdopen));"""
    ...
  def feof(arg: ffi.CData) -> int:
    """int	 feof(FILE *);"""
    ...
  def ferror(arg: ffi.CData) -> int:
    """int	 ferror(FILE *);"""
    ...
  def fflush(arg: ffi.CData) -> int:
    """int	 fflush(FILE *);"""
    ...
  def fgetc(arg: ffi.CData) -> int:
    """int	 fgetc(FILE *);"""
    ...
  def fgetln(arg: ffi.CData, arg2: ffi.CData) -> ffi.CData:
    """char	*fgetln(FILE *, size_t *);"""
    ...
  def fgetpos(arg: ffi.CData, arg2: ffi.CData) -> int:
    """int	 fgetpos(FILE * __restrict, fpos_t *);"""
    ...
  def fgets(arg: ffi.CData, arg2: int, arg3: ffi.CData) -> ffi.CData:
    """char	*fgets(char * __restrict, int, FILE *);"""
    ...
  def fileno(arg: ffi.CData) -> int:
    """int	 fileno(FILE *);"""
    ...
  def flockfile(arg: ffi.CData) -> None:
    """void	 flockfile(FILE *);"""
    ...
  def fmemopen(__buf: ffi.CData, __size: int, __mode: ffi.CData) -> ffi.CData:
    """FILE *fmemopen(void * __restrict __buf, size_t __size, const char * __restrict __mode) __API_AVAILABLE(macos(10.13), ios(11.0), tvos(11.0), watchos(4.0));"""
    ...
  def fmtcheck(arg: ffi.CData, arg2: ffi.CData) -> ffi.CData:
    """__const char *fmtcheck(const char *, const char *) __attribute__((format_arg(2)));"""
    ...
  def fopen(__filename: ffi.CData, __mode: ffi.CData) -> ffi.CData:
    """FILE	*fopen(const char * __restrict __filename, const char * __restrict __mode) __DARWIN_ALIAS_STARTING(__MAC_10_6, __IPHONE_2_0, __DARWIN_ALIAS(fopen));"""
    ...
  def fprintf(arg: ffi.CData, arg2: ffi.CData, *args2) -> int:
    """int	 fprintf(FILE * __restrict, const char * __restrict, ...) __printflike(2, 3);"""
    ...
  def fpurge(arg: ffi.CData) -> int:
    """int	 fpurge(FILE *);"""
    ...
  def fputc(arg: int, arg2: ffi.CData) -> int:
    """int	 fputc(int, FILE *);"""
    ...
  def fputs(arg: ffi.CData, arg2: ffi.CData) -> int:
    """int	 fputs(const char * __restrict, FILE * __restrict) __DARWIN_ALIAS(fputs);"""
    ...
  def fread(__ptr: ffi.CData, __size: int, __nitems: int, __stream: ffi.CData) -> int:
    """size_t	 fread(void * __restrict __ptr, size_t __size, size_t __nitems, FILE * __restrict __stream);"""
    ...
  def freopen(arg: ffi.CData, arg2: ffi.CData, arg3: ffi.CData) -> ffi.CData:
    """
    FILE	*freopen(const char * __restrict, const char * __restrict,
                     FILE * __restrict) __DARWIN_ALIAS(freopen);
    """
    ...
  def fscanf(arg: ffi.CData, arg2: ffi.CData, *args2) -> int:
    """int	 fscanf(FILE * __restrict, const char * __restrict, ...) __scanflike(2, 3);"""
    ...
  def fseek(arg: ffi.CData, arg2: int, arg3: int) -> int:
    """int	 fseek(FILE *, long, int);"""
    ...
  def fseeko(__stream: ffi.CData, __offset: ffi.CData, __whence: int) -> int:
    """int	 fseeko(FILE * __stream, off_t __offset, int __whence);"""
    ...
  def fsetpos(arg: ffi.CData, arg2: ffi.CData) -> int:
    """int	 fsetpos(FILE *, const fpos_t *);"""
    ...
  def ftell(arg: ffi.CData) -> int:
    """long	 ftell(FILE *);"""
    ...
  def ftello(__stream: ffi.CData) -> ffi.CData:
    """off_t	 ftello(FILE * __stream);"""
    ...
  def ftrylockfile(arg: ffi.CData) -> int:
    """int	 ftrylockfile(FILE *);"""
    ...
  def funlockfile(arg: ffi.CData) -> None:
    """void	 funlockfile(FILE *);"""
    ...
  def funopen(arg: ffi.CData, arg2: ffi.CData, arg3: ffi.CData, arg4: ffi.CData, arg5: ffi.CData) -> ffi.CData:
    """
    FILE	*funopen(const void *,
                     int (* _Nullable)(void *, char *, int),
                     int (* _Nullable)(void *, const char *, int),
                     fpos_t (* _Nullable)(void *, fpos_t, int),
                     int (* _Nullable)(void *));
    """
    ...
  def fwrite(__ptr: ffi.CData, __size: int, __nitems: int, __stream: ffi.CData) -> int:
    """size_t	 fwrite(const void * __restrict __ptr, size_t __size, size_t __nitems, FILE * __restrict __stream) __DARWIN_ALIAS(fwrite);"""
    ...
  def getc(arg: ffi.CData) -> int:
    """int	 getc(FILE *);"""
    ...
  def getc_unlocked(arg: ffi.CData) -> int:
    """int	 getc_unlocked(FILE *);"""
    ...
  def getchar() -> int:
    """int	 getchar(void);"""
    ...
  def getchar_unlocked() -> int:
    """int	 getchar_unlocked(void);"""
    ...
  def getdelim(__linep: ffi.CData, __linecapp: ffi.CData, __delimiter: int, __stream: ffi.CData) -> ffi.CData:
    """ssize_t getdelim(char ** __restrict __linep, size_t * __restrict __linecapp, int __delimiter, FILE * __restrict __stream) __OSX_AVAILABLE_STARTING(__MAC_10_7, __IPHONE_4_3);"""
    ...
  def getline(__linep: ffi.CData, __linecapp: ffi.CData, __stream: ffi.CData) -> ffi.CData:
    """ssize_t getline(char ** __restrict __linep, size_t * __restrict __linecapp, FILE * __restrict __stream) __OSX_AVAILABLE_STARTING(__MAC_10_7, __IPHONE_4_3);"""
    ...
  def gets(arg: ffi.CData) -> ffi.CData:
    """char	*gets(char *);"""
    ...
  def getw(arg: ffi.CData) -> int:
    """int	 getw(FILE *);"""
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
  def ggml_are_same_shape(t0: ffi.CData, t1: ffi.CData) -> bool:
    """    GGML_API bool ggml_are_same_shape (const struct ggml_tensor * t0, const struct ggml_tensor * t1);"""
    ...
  def ggml_are_same_stride(t0: ffi.CData, t1: ffi.CData) -> bool:
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
  def ggml_backend_buffer_is_host(buffer: ffi.CData) -> bool:
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
  def ggml_backend_buft_is_host(buft: ffi.CData) -> bool:
    """    GGML_API           bool                  ggml_backend_buft_is_host         (ggml_backend_buffer_type_t buft);"""
    ...
  def ggml_backend_buft_name(buft: ffi.CData) -> ffi.CData:
    """
    buffer type

        GGML_API           const char *          ggml_backend_buft_name            (ggml_backend_buffer_type_t buft);
    """
    ...
  def ggml_backend_buft_supports_backend(buft: ffi.CData, backend: ffi.CData) -> bool:
    """    GGML_API           bool                  ggml_backend_buft_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend);"""
    ...
  def ggml_backend_compare_graph_backend(backend1: ffi.CData, backend2: ffi.CData, graph: ffi.CData, callback: ffi.CData, user_data: ffi.CData) -> bool:
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
  def ggml_backend_cuda_register_host_buffer(buffer: ffi.CData, size: int) -> bool:
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
  def ggml_backend_is_cpu(backend: ffi.CData) -> bool:
    """    GGML_API GGML_CALL bool ggml_backend_is_cpu                (ggml_backend_t backend);"""
    ...
  def ggml_backend_is_cuda(backend: ffi.CData) -> bool:
    """GGML_API GGML_CALL bool ggml_backend_is_cuda(ggml_backend_t backend);"""
    ...
  def ggml_backend_is_kompute(backend: ffi.CData) -> bool:
    """GGML_API bool ggml_backend_is_kompute(ggml_backend_t backend);"""
    ...
  def ggml_backend_is_metal(backend: ffi.CData) -> bool:
    """GGML_API bool ggml_backend_is_metal(ggml_backend_t backend);"""
    ...
  def ggml_backend_is_rpc(backend: ffi.CData) -> bool:
    """GGML_API GGML_CALL bool ggml_backend_is_rpc(ggml_backend_t backend);"""
    ...
  def ggml_backend_is_vk(backend: ffi.CData) -> bool:
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
  def ggml_backend_metal_supports_family(backend: ffi.CData, family: int) -> bool:
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
  def ggml_backend_offload_op(backend: ffi.CData, op: ffi.CData) -> bool:
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
  def ggml_backend_sched_alloc_graph(sched: ffi.CData, graph: ffi.CData) -> bool:
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
  def ggml_backend_sched_new(backends: ffi.CData, bufts: ffi.CData, n_backends: int, graph_size: int, parallel: bool) -> ffi.CData:
    """
    Initialize a backend scheduler

        GGML_API ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t * backends, ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size, bool parallel);
    """
    ...
  def ggml_backend_sched_reserve(sched: ffi.CData, measure_graph: ffi.CData) -> bool:
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
  def ggml_backend_supports_op(backend: ffi.CData, op: ffi.CData) -> bool:
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
  def ggml_build_backward_expand(ctx: ffi.CData, gf: ffi.CData, gb: ffi.CData, keep: bool) -> None:
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
  def ggml_cl_can_mul_mat(src0: ffi.CData, src1: ffi.CData, dst: ffi.CData) -> bool:
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
  def ggml_flash_attn(ctx: ffi.CData, q: ffi.CData, k: ffi.CData, v: ffi.CData, masked: bool) -> ffi.CData:
    """
        GGML_API struct ggml_tensor * ggml_flash_attn(
                struct ggml_context * ctx,
                struct ggml_tensor  * q,
                struct ggml_tensor  * k,
                struct ggml_tensor  * v,
                bool                  masked);
    """
    ...
  def ggml_flash_attn_back(ctx: ffi.CData, q: ffi.CData, k: ffi.CData, v: ffi.CData, d: ffi.CData, masked: bool) -> ffi.CData:
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
  def ggml_gallocr_alloc_graph(galloc: ffi.CData, graph: ffi.CData) -> bool:
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
  def ggml_gallocr_reserve(galloc: ffi.CData, graph: ffi.CData) -> bool:
    """
    pre-allocate buffers from a measure graph - does not allocate or modify the graph
    call with a worst-case graph to avoid buffer reallocations
    not strictly required for single buffer usage: ggml_gallocr_alloc_graph will reallocate the buffers automatically if needed
    returns false if the buffer allocation failed

    GGML_API bool ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
    """
    ...
  def ggml_gallocr_reserve_n(galloc: ffi.CData, graph: ffi.CData, node_buffer_ids: ffi.CData, leaf_buffer_ids: ffi.CData) -> bool:
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
  def ggml_get_no_alloc(ctx: ffi.CData) -> bool:
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
  def ggml_graph_overhead_custom(size: int, grads: bool) -> int:
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
  def ggml_guid_matches(guid_a: ffi.CData, guid_b: ffi.CData) -> bool:
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
  def ggml_im2col(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, s0: int, s1: int, p0: int, p1: int, d0: int, d1: int, is_2D: bool, dst_type: int) -> ffi.CData:
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
  def ggml_is_3d(tensor: ffi.CData) -> bool:
    """    GGML_API           bool ggml_is_3d        (const struct ggml_tensor * tensor);"""
    ...
  def ggml_is_contiguous(tensor: ffi.CData) -> bool:
    """    GGML_API GGML_CALL bool ggml_is_contiguous(const struct ggml_tensor * tensor);"""
    ...
  def ggml_is_empty(tensor: ffi.CData) -> bool:
    """    GGML_API GGML_CALL bool ggml_is_empty     (const struct ggml_tensor * tensor);"""
    ...
  def ggml_is_matrix(tensor: ffi.CData) -> bool:
    """    GGML_API           bool ggml_is_matrix    (const struct ggml_tensor * tensor);"""
    ...
  def ggml_is_numa() -> bool:
    """    GGML_API bool    ggml_is_numa(void); // true if init detected that system has >1 NUMA node"""
    ...
  def ggml_is_permuted(tensor: ffi.CData) -> bool:
    """    GGML_API GGML_CALL bool ggml_is_permuted  (const struct ggml_tensor * tensor);"""
    ...
  def ggml_is_quantized(type: int) -> bool:
    """    GGML_API GGML_CALL bool    ggml_is_quantized(enum ggml_type type);"""
    ...
  def ggml_is_scalar(tensor: ffi.CData) -> bool:
    """    GGML_API           bool ggml_is_scalar    (const struct ggml_tensor * tensor);"""
    ...
  def ggml_is_transposed(tensor: ffi.CData) -> bool:
    """    GGML_API GGML_CALL bool ggml_is_transposed(const struct ggml_tensor * tensor);"""
    ...
  def ggml_is_vector(tensor: ffi.CData) -> bool:
    """    GGML_API           bool ggml_is_vector    (const struct ggml_tensor * tensor);"""
    ...
  def ggml_leaky_relu(ctx: ffi.CData, a: ffi.CData, negative_slope: float, inplace: bool) -> ffi.CData:
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
  def ggml_mpi_backend_free() -> None:
    """void ggml_mpi_backend_free(void);"""
    ...
  def ggml_mpi_backend_init() -> None:
    """void ggml_mpi_backend_init(void);"""
    ...
  def ggml_mpi_eval_init(ctx_mpi: ffi.CData, n_tokens: ffi.CData, n_past: ffi.CData, n_threads: ffi.CData) -> None:
    """
    void ggml_mpi_eval_init(
            struct ggml_mpi_context * ctx_mpi,
                                int * n_tokens,
                                int * n_past,
                                int * n_threads);
    """
    ...
  def ggml_mpi_free(ctx: ffi.CData) -> None:
    """void ggml_mpi_free(struct ggml_mpi_context * ctx);"""
    ...
  def ggml_mpi_graph_compute_post(ctx_mpi: ffi.CData, gf: ffi.CData, n_layers: int) -> None:
    """
    void ggml_mpi_graph_compute_post(
            struct ggml_mpi_context * ctx_mpi,
                 struct ggml_cgraph * gf,
                                int   n_layers);
    """
    ...
  def ggml_mpi_graph_compute_pre(ctx_mpi: ffi.CData, gf: ffi.CData, n_layers: int) -> None:
    """
    void ggml_mpi_graph_compute_pre(
            struct ggml_mpi_context * ctx_mpi,
                 struct ggml_cgraph * gf,
                                int   n_layers);
    """
    ...
  def ggml_mpi_init() -> ffi.CData:
    """struct ggml_mpi_context * ggml_mpi_init(void);"""
    ...
  def ggml_mpi_rank(ctx: ffi.CData) -> int:
    """int ggml_mpi_rank(struct ggml_mpi_context * ctx);"""
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
  def ggml_new_graph_custom(ctx: ffi.CData, size: int, grads: bool) -> ffi.CData:
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
  def ggml_quantize_requires_imatrix(type: int) -> bool:
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
  def ggml_rope_back(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, n_dims: int, mode: int, n_ctx: int, n_orig_ctx: int, freq_base: float, freq_scale: float, ext_factor: float, attn_factor: float, beta_fast: float, beta_slow: float, xpos_base: float, xpos_down: bool) -> ffi.CData:
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
  def ggml_rope_xpos_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, n_dims: int, base: float, down: bool) -> ffi.CData:
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
  def ggml_set_no_alloc(ctx: ffi.CData, no_alloc: bool) -> None:
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
  def ggml_validate_row_data(type: int, data: ffi.CData, nbytes: int) -> bool:
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
  def ggml_vk_get_device(device: ffi.CData, memoryRequired: int, name: ffi.CData) -> bool:
    """bool ggml_vk_get_device(struct ggml_vk_device * device, size_t memoryRequired, const char * name);"""
    ...
  def ggml_vk_has_device() -> bool:
    """bool ggml_vk_has_device(void);"""
    ...
  def ggml_vk_has_vulkan() -> bool:
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
  def gguf_get_val_bool(ctx: ffi.CData, key_id: int) -> bool:
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
  def gguf_set_val_bool(ctx: ffi.CData, key: ffi.CData, val: bool) -> None:
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
  def gguf_write_to_file(ctx: ffi.CData, fname: ffi.CData, only_meta: bool) -> None:
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
  def llama_add_bos_token(model: ffi.CData) -> int:
    """
    Returns -1 if unknown, 1 for true or 0 for false.

        LLAMA_API int32_t         llama_add_bos_token(const struct llama_model * model);
    """
    ...
  def llama_add_eos_token(model: ffi.CData) -> int:
    """
    Returns -1 if unknown, 1 for true or 0 for false.

        LLAMA_API int32_t         llama_add_eos_token(const struct llama_model * model);
    """
    ...
  def llama_backend_free() -> None:
    """
    Call once at the end of the program - currently only used for MPI

        LLAMA_API void llama_backend_free(void);
    """
    ...
  def llama_backend_init() -> None:
    """
    Initialize the llama + ggml backend
    If numa is true, use NUMA optimizations
    Call once at the start of the program

        LLAMA_API void llama_backend_init(void);
    """
    ...
  def llama_batch_free(batch: ffi.CData) -> None:
    """
    Frees a batch of tokens allocated with llama_batch_init()

        LLAMA_API void llama_batch_free(struct llama_batch batch);
    """
    ...
  def llama_batch_get_one(tokens: ffi.CData, n_tokens: int, pos_0: ffi.CData, seq_id: ffi.CData) -> ffi.CData:
    """
    Return batch for single sequence of tokens starting at pos_0
    
    NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
    

        LLAMA_API struct llama_batch llama_batch_get_one(
                      llama_token * tokens,
                          int32_t   n_tokens,
                        llama_pos   pos_0,
                     llama_seq_id   seq_id);
    """
    ...
  def llama_batch_init(n_tokens: int, embd: int, n_seq_max: int) -> ffi.CData:
    """
    Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
    Each token can be assigned up to n_seq_max sequence ids
    The batch has to be freed with llama_batch_free()
    If embd != 0, llama_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
    Otherwise, llama_batch.token will be allocated to store n_tokens llama_token
    The rest of the llama_batch members are allocated with size n_tokens
    All members are left uninitialized

        LLAMA_API struct llama_batch llama_batch_init(
                int32_t n_tokens,
                int32_t embd,
                int32_t n_seq_max);
    """
    ...
  def llama_beam_search(ctx: ffi.CData, callback: ffi.CData, callback_data: ffi.CData, n_beams: int, n_past: int, n_predict: int) -> None:
    """
    @details Deterministically returns entire sentence constructed by a beam search.
    @param ctx Pointer to the llama_context.
    @param callback Invoked for each iteration of the beam_search loop, passing in beams_state.
    @param callback_data A pointer that is simply passed back to callback.
    @param n_beams Number of beams to use.
    @param n_past Number of tokens already evaluated.
    @param n_predict Maximum number of tokens to predict. EOS may occur earlier.

        LLAMA_API void llama_beam_search(
                       struct llama_context * ctx,
            llama_beam_search_callback_fn_t   callback,
                                       void * callback_data,
                                     size_t   n_beams,
                                    int32_t   n_past,
                                    int32_t   n_predict);
    """
    ...
  def llama_chat_apply_template(model: ffi.CData, tmpl: ffi.CData, chat: ffi.CData, n_msg: int, add_ass: bool, buf: ffi.CData, length: int) -> int:
    """
    Apply chat template. Inspired by hf apply_chat_template() on python.
    Both "model" and "custom_template" are optional, but at least one is required. "custom_template" has higher precedence than "model"
    NOTE: This function does not use a jinja parser. It only support a pre-defined list of template. See more: https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
    @param tmpl A Jinja template to use for this chat. If this is nullptr, the model’s default chat template will be used instead.
    @param chat Pointer to a list of multiple llama_chat_message
    @param n_msg Number of llama_chat_message in this chat
    @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
    @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
    @param length The size of the allocated buffer
    @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.

        LLAMA_API int32_t llama_chat_apply_template(
                  const struct llama_model * model,
                                const char * tmpl,
           const struct llama_chat_message * chat,
                                    size_t   n_msg,
                                      bool   add_ass,
                                      char * buf,
                                   int32_t   length);
    """
    ...
  def llama_context_default_params() -> ffi.CData:
    """    LLAMA_API struct llama_context_params llama_context_default_params(void);"""
    ...
  def llama_control_vector_apply(lctx: ffi.CData, data: ffi.CData, len: int, n_embd: int, il_start: int, il_end: int) -> int:
    """
    Apply a loaded control vector to a llama_context, or if data is NULL, clear
    the currently loaded vector.
    n_embd should be the size of a single layer's control, and data should point
    to an n_embd x n_layers buffer starting from layer 1.
    il_start and il_end are the layer range the vector should apply to (both inclusive)
    See llama_control_vector_load in common to load a control vector.

        LLAMA_API int32_t llama_control_vector_apply(
                struct llama_context * lctx,
                         const float * data,
                              size_t   len,
                             int32_t   n_embd,
                             int32_t   il_start,
                             int32_t   il_end);
    """
    ...
  def llama_copy_state_data(ctx: ffi.CData, dst: ffi.CData) -> int:
    """
        LLAMA_API DEPRECATED(size_t llama_copy_state_data(
                struct llama_context * ctx,
                             uint8_t * dst),
            "use llama_state_get_data instead");
    """
    ...
  def llama_decode(ctx: ffi.CData, batch: ffi.CData) -> int:
    """
    Positive return values does not mean a fatal error, but rather a warning.
    0 - success
    1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
    < 0 - error

        LLAMA_API int32_t llama_decode(
                struct llama_context * ctx,
                  struct llama_batch   batch);
    """
    ...
  def llama_dump_timing_info_yaml(stream: ffi.CData, ctx: ffi.CData) -> None:
    """    LLAMA_API void llama_dump_timing_info_yaml(FILE * stream, const struct llama_context * ctx);"""
    ...
  def llama_free(ctx: ffi.CData) -> None:
    """
    Frees all allocated memory

        LLAMA_API void llama_free(struct llama_context * ctx);
    """
    ...
  def llama_free_model(model: ffi.CData) -> None:
    """    LLAMA_API void llama_free_model(struct llama_model * model);"""
    ...
  def llama_get_embeddings(ctx: ffi.CData) -> ffi.CData:
    """
    Get all output token embeddings.
    when pooling_type == LLAMA_POOLING_TYPE_NONE or when using a generative model,
    the embeddings for which llama_batch.logits[i] != 0 are stored contiguously
    in the order they have appeared in the batch.
    shape: [n_outputs*n_embd]
    Otherwise, returns NULL.

        LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
    """
    ...
  def llama_get_embeddings_ith(ctx: ffi.CData, i: int) -> ffi.CData:
    """
    Get the embeddings for the ith token. For positive indices, Equivalent to:
    llama_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
    Negative indicies can be used to access embeddings in reverse order, -1 is the last embedding.
    shape: [n_embd] (1-dimensional)
    returns NULL for invalid ids.

        LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
    """
    ...
  def llama_get_embeddings_seq(ctx: ffi.CData, seq_id: ffi.CData) -> ffi.CData:
    """
    Get the embeddings for a sequence id
    Returns NULL if pooling_type is LLAMA_POOLING_TYPE_NONE
    shape: [n_embd] (1-dimensional)

        LLAMA_API float * llama_get_embeddings_seq(struct llama_context * ctx, llama_seq_id seq_id);
    """
    ...
  def llama_get_kv_cache_token_count(ctx: ffi.CData) -> int:
    """
    Returns the number of tokens in the KV cache (slow, use only for debug)
    If a KV cell has multiple sequences assigned to it, it will be counted multiple times

        LLAMA_API int32_t llama_get_kv_cache_token_count(const struct llama_context * ctx);
    """
    ...
  def llama_get_kv_cache_used_cells(ctx: ffi.CData) -> int:
    """
    Returns the number of used KV cells (i.e. have at least one sequence assigned to them)

        LLAMA_API int32_t llama_get_kv_cache_used_cells(const struct llama_context * ctx);
    """
    ...
  def llama_get_logits(ctx: ffi.CData) -> ffi.CData:
    """
    Token logits obtained from the last call to llama_decode()
    The logits for which llama_batch.logits[i] != 0 are stored contiguously
    in the order they have appeared in the batch.
    Rows: number of tokens for which llama_batch.logits[i] != 0
    Cols: n_vocab

        LLAMA_API float * llama_get_logits(struct llama_context * ctx);
    """
    ...
  def llama_get_logits_ith(ctx: ffi.CData, i: int) -> ffi.CData:
    """
    Logits for the ith token. For positive indices, Equivalent to:
    llama_get_logits(ctx) + ctx->output_ids[i]*n_vocab
    Negative indicies can be used to access logits in reverse order, -1 is the last logit.
    returns NULL for invalid ids.

        LLAMA_API float * llama_get_logits_ith(struct llama_context * ctx, int32_t i);
    """
    ...
  def llama_get_model(ctx: ffi.CData) -> ffi.CData:
    """    LLAMA_API const struct llama_model * llama_get_model(const struct llama_context * ctx);"""
    ...
  def llama_get_model_tensor(model: ffi.CData, name: ffi.CData) -> ffi.CData:
    """
    Get a llama model tensor

        LLAMA_API struct ggml_tensor * llama_get_model_tensor(struct llama_model * model, const char * name);
    """
    ...
  def llama_get_state_size(ctx: ffi.CData) -> int:
    """
        LLAMA_API DEPRECATED(size_t llama_get_state_size(const struct llama_context * ctx),
            "use llama_state_get_size instead");
    """
    ...
  def llama_get_timings(ctx: ffi.CData) -> ffi.CData:
    """
    Performance information

        LLAMA_API struct llama_timings llama_get_timings(struct llama_context * ctx);
    """
    ...
  def llama_grammar_accept_token(ctx: ffi.CData, grammar: ffi.CData, token: ffi.CData) -> None:
    """
    @details Accepts the sampled token into the grammar

        LLAMA_API void llama_grammar_accept_token(
                struct llama_context * ctx,
                struct llama_grammar * grammar,
                         llama_token   token);
    """
    ...
  def llama_grammar_copy(grammar: ffi.CData) -> ffi.CData:
    """    LLAMA_API struct llama_grammar * llama_grammar_copy(const struct llama_grammar * grammar);"""
    ...
  def llama_grammar_free(grammar: ffi.CData) -> None:
    """    LLAMA_API void llama_grammar_free(struct llama_grammar * grammar);"""
    ...
  def llama_grammar_init(rules: ffi.CData, n_rules: int, start_rule_index: int) -> ffi.CData:
    """
        LLAMA_API struct llama_grammar * llama_grammar_init(
                const llama_grammar_element ** rules,
                                     size_t    n_rules,
                                     size_t    start_rule_index);
    """
    ...
  def llama_kv_cache_clear(ctx: ffi.CData) -> None:
    """
    Clear the KV cache - both cell info is erased and KV data is zeroed

        LLAMA_API void llama_kv_cache_clear(
                struct llama_context * ctx);
    """
    ...
  def llama_kv_cache_defrag(ctx: ffi.CData) -> None:
    """
    Defragment the KV cache
    This will be applied:
    - lazily on next llama_decode()
    - explicitly with llama_kv_cache_update()

        LLAMA_API void llama_kv_cache_defrag(struct llama_context * ctx);
    """
    ...
  def llama_kv_cache_seq_add(ctx: ffi.CData, seq_id: ffi.CData, p0: ffi.CData, p1: ffi.CData, delta: ffi.CData) -> None:
    """
    Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
    If the KV cache is RoPEd, the KV data is updated accordingly:
    - lazily on next llama_decode()
    - explicitly with llama_kv_cache_update()
    p0 < 0 : [0,  p1]
    p1 < 0 : [p0, inf)

        LLAMA_API void llama_kv_cache_seq_add(
                struct llama_context * ctx,
                        llama_seq_id   seq_id,
                           llama_pos   p0,
                           llama_pos   p1,
                           llama_pos   delta);
    """
    ...
  def llama_kv_cache_seq_cp(ctx: ffi.CData, seq_id_src: ffi.CData, seq_id_dst: ffi.CData, p0: ffi.CData, p1: ffi.CData) -> None:
    """
    Copy all tokens that belong to the specified sequence to another sequence
    Note that this does not allocate extra KV cache memory - it simply assigns the tokens to the new sequence
    p0 < 0 : [0,  p1]
    p1 < 0 : [p0, inf)

        LLAMA_API void llama_kv_cache_seq_cp(
                struct llama_context * ctx,
                        llama_seq_id   seq_id_src,
                        llama_seq_id   seq_id_dst,
                           llama_pos   p0,
                           llama_pos   p1);
    """
    ...
  def llama_kv_cache_seq_div(ctx: ffi.CData, seq_id: ffi.CData, p0: ffi.CData, p1: ffi.CData, d: int) -> None:
    """
    Integer division of the positions by factor of `d > 1`
    If the KV cache is RoPEd, the KV data is updated accordingly:
    - lazily on next llama_decode()
    - explicitly with llama_kv_cache_update()
    p0 < 0 : [0,  p1]
    p1 < 0 : [p0, inf)

        LLAMA_API void llama_kv_cache_seq_div(
                struct llama_context * ctx,
                        llama_seq_id   seq_id,
                           llama_pos   p0,
                           llama_pos   p1,
                                 int   d);
    """
    ...
  def llama_kv_cache_seq_keep(ctx: ffi.CData, seq_id: ffi.CData) -> None:
    """
    Removes all tokens that do not belong to the specified sequence

        LLAMA_API void llama_kv_cache_seq_keep(
                struct llama_context * ctx,
                        llama_seq_id   seq_id);
    """
    ...
  def llama_kv_cache_seq_pos_max(ctx: ffi.CData, seq_id: ffi.CData) -> ffi.CData:
    """
    Returns the largest position present in the KV cache for the specified sequence

        LLAMA_API llama_pos llama_kv_cache_seq_pos_max(
                struct llama_context * ctx,
                        llama_seq_id   seq_id);
    """
    ...
  def llama_kv_cache_seq_rm(ctx: ffi.CData, seq_id: ffi.CData, p0: ffi.CData, p1: ffi.CData) -> bool:
    """
    Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
    Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
    seq_id < 0 : match any sequence
    p0 < 0     : [0,  p1]
    p1 < 0     : [p0, inf)

        LLAMA_API bool llama_kv_cache_seq_rm(
                struct llama_context * ctx,
                        llama_seq_id   seq_id,
                           llama_pos   p0,
                           llama_pos   p1);
    """
    ...
  def llama_kv_cache_update(ctx: ffi.CData) -> None:
    """
    Apply the KV cache updates (such as K-shifts, defragmentation, etc.)

        LLAMA_API void llama_kv_cache_update(struct llama_context * ctx);
    """
    ...
  def llama_kv_cache_view_free(view: ffi.CData) -> None:
    """
    Free a KV cache view. (use only for debugging purposes)

        LLAMA_API void llama_kv_cache_view_free(struct llama_kv_cache_view * view);
    """
    ...
  def llama_kv_cache_view_init(ctx: ffi.CData, n_seq_max: int) -> ffi.CData:
    """
    Create an empty KV cache view. (use only for debugging purposes)

        LLAMA_API struct llama_kv_cache_view llama_kv_cache_view_init(const struct llama_context * ctx, int32_t n_seq_max);
    """
    ...
  def llama_kv_cache_view_update(ctx: ffi.CData, view: ffi.CData) -> None:
    """
    Update the KV cache view structure with the current state of the KV cache. (use only for debugging purposes)

        LLAMA_API void llama_kv_cache_view_update(const struct llama_context * ctx, struct llama_kv_cache_view * view);
    """
    ...
  def llama_load_model_from_file(path_model: ffi.CData, params: ffi.CData) -> ffi.CData:
    """
        LLAMA_API struct llama_model * llama_load_model_from_file(
                                 const char * path_model,
                struct llama_model_params     params);
    """
    ...
  def llama_load_session_file(ctx: ffi.CData, path_session: ffi.CData, tokens_out: ffi.CData, n_token_capacity: int, n_token_count_out: ffi.CData) -> bool:
    """
        LLAMA_API DEPRECATED(bool llama_load_session_file(
                struct llama_context * ctx,
                          const char * path_session,
                         llama_token * tokens_out,
                              size_t   n_token_capacity,
                              size_t * n_token_count_out),
            "use llama_state_load_file instead");
    """
    ...
  def llama_log_set(log_callback: ffi.CData, user_data: ffi.CData) -> None:
    """
    Set callback for all future logging events.
    If this is not called, or NULL is supplied, everything is output on stderr.

        LLAMA_API void llama_log_set(ggml_log_callback log_callback, void * user_data);
    """
    ...
  def llama_max_devices() -> int:
    """    LLAMA_API size_t llama_max_devices(void);"""
    ...
  def llama_model_apply_lora_from_file(model: ffi.CData, path_lora: ffi.CData, scale: float, path_base_model: ffi.CData, n_threads: int) -> int:
    """
    Apply a LoRA adapter to a loaded model
    path_base_model is the path to a higher quality model to use as a base for
    the layers modified by the adapter. Can be NULL to use the current loaded model.
    The model needs to be reloaded before applying a new adapter, otherwise the adapter
    will be applied on top of the previous one
    Returns 0 on success

        LLAMA_API int32_t llama_model_apply_lora_from_file(
                const struct llama_model * model,
                              const char * path_lora,
                                   float   scale,
                              const char * path_base_model,
                                 int32_t   n_threads);
    """
    ...
  def llama_model_default_params() -> ffi.CData:
    """
    Helpers for getting default parameters

        LLAMA_API struct llama_model_params llama_model_default_params(void);
    """
    ...
  def llama_model_desc(model: ffi.CData, buf: ffi.CData, buf_size: int) -> int:
    """
    Get a string describing the model type

        LLAMA_API int32_t llama_model_desc(const struct llama_model * model, char * buf, size_t buf_size);
    """
    ...
  def llama_model_meta_count(model: ffi.CData) -> int:
    """
    Get the number of metadata key/value pairs

        LLAMA_API int32_t llama_model_meta_count(const struct llama_model * model);
    """
    ...
  def llama_model_meta_key_by_index(model: ffi.CData, i: int, buf: ffi.CData, buf_size: int) -> int:
    """
    Get metadata key name by index

        LLAMA_API int32_t llama_model_meta_key_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size);
    """
    ...
  def llama_model_meta_val_str(model: ffi.CData, key: ffi.CData, buf: ffi.CData, buf_size: int) -> int:
    """
    Get metadata value as a string by key name

        LLAMA_API int32_t llama_model_meta_val_str(const struct llama_model * model, const char * key, char * buf, size_t buf_size);
    """
    ...
  def llama_model_meta_val_str_by_index(model: ffi.CData, i: int, buf: ffi.CData, buf_size: int) -> int:
    """
    Get metadata value as a string by index

        LLAMA_API int32_t llama_model_meta_val_str_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size);
    """
    ...
  def llama_model_n_params(model: ffi.CData) -> int:
    """
    Returns the total number of parameters in the model

        LLAMA_API uint64_t llama_model_n_params(const struct llama_model * model);
    """
    ...
  def llama_model_quantize(fname_inp: ffi.CData, fname_out: ffi.CData, params: ffi.CData) -> int:
    """
    Returns 0 on success

        LLAMA_API uint32_t llama_model_quantize(
                const char * fname_inp,
                const char * fname_out,
                const llama_model_quantize_params * params);
    """
    ...
  def llama_model_quantize_default_params() -> ffi.CData:
    """    LLAMA_API struct llama_model_quantize_params llama_model_quantize_default_params(void);"""
    ...
  def llama_model_size(model: ffi.CData) -> int:
    """
    Returns the total size of all the tensors in the model in bytes

        LLAMA_API uint64_t llama_model_size(const struct llama_model * model);
    """
    ...
  def llama_n_batch(ctx: ffi.CData) -> int:
    """    LLAMA_API uint32_t llama_n_batch    (const struct llama_context * ctx);"""
    ...
  def llama_n_ctx(ctx: ffi.CData) -> int:
    """    LLAMA_API uint32_t llama_n_ctx      (const struct llama_context * ctx);"""
    ...
  def llama_n_ctx_train(model: ffi.CData) -> int:
    """    LLAMA_API int32_t llama_n_ctx_train(const struct llama_model * model);"""
    ...
  def llama_n_embd(model: ffi.CData) -> int:
    """    LLAMA_API int32_t llama_n_embd     (const struct llama_model * model);"""
    ...
  def llama_n_layer(model: ffi.CData) -> int:
    """    LLAMA_API int32_t llama_n_layer    (const struct llama_model * model);"""
    ...
  def llama_n_seq_max(ctx: ffi.CData) -> int:
    """    LLAMA_API uint32_t llama_n_seq_max  (const struct llama_context * ctx);"""
    ...
  def llama_n_ubatch(ctx: ffi.CData) -> int:
    """    LLAMA_API uint32_t llama_n_ubatch   (const struct llama_context * ctx);"""
    ...
  def llama_n_vocab(model: ffi.CData) -> int:
    """    LLAMA_API int32_t llama_n_vocab    (const struct llama_model * model);"""
    ...
  def llama_new_context_with_model(model: ffi.CData, params: ffi.CData) -> ffi.CData:
    """
        LLAMA_API struct llama_context * llama_new_context_with_model(
                         struct llama_model * model,
                struct llama_context_params   params);
    """
    ...
  def llama_numa_init(numa: int) -> None:
    """
    optional:

        LLAMA_API void llama_numa_init(enum ggml_numa_strategy numa);
    """
    ...
  def llama_pooling_type(ctx: ffi.CData) -> int:
    """    LLAMA_API enum llama_pooling_type llama_pooling_type(const struct llama_context * ctx);"""
    ...
  def llama_print_system_info() -> ffi.CData:
    """
    Print system information

        LLAMA_API const char * llama_print_system_info(void);
    """
    ...
  def llama_print_timings(ctx: ffi.CData) -> None:
    """    LLAMA_API void llama_print_timings(struct llama_context * ctx);"""
    ...
  def llama_reset_timings(ctx: ffi.CData) -> None:
    """    LLAMA_API void llama_reset_timings(struct llama_context * ctx);"""
    ...
  def llama_rope_freq_scale_train(model: ffi.CData) -> float:
    """
    Get the model's RoPE frequency scaling factor

        LLAMA_API float llama_rope_freq_scale_train(const struct llama_model * model);
    """
    ...
  def llama_rope_type(model: ffi.CData) -> int:
    """    LLAMA_API enum llama_rope_type    llama_rope_type   (const struct llama_model   * model);"""
    ...
  def llama_sample_apply_guidance(ctx: ffi.CData, logits: ffi.CData, logits_guidance: ffi.CData, scale: float) -> None:
    """
    @details Apply classifier-free guidance to the logits as described in academic paper "Stay on topic with Classifier-Free Guidance" https://arxiv.org/abs/2306.17806
    @param logits Logits extracted from the original generation context.
    @param logits_guidance Logits extracted from a separate context from the same model. Other than a negative prompt at the beginning, it should have all generated and user input tokens copied from the main context.
    @param scale Guidance strength. 1.0f means no guidance. Higher values mean stronger guidance.

        LLAMA_API void llama_sample_apply_guidance(
                  struct llama_context * ctx,
                                 float * logits,
                                 float * logits_guidance,
                                 float   scale);
    """
    ...
  def llama_sample_entropy(ctx: ffi.CData, candidates_p: ffi.CData, min_temp: float, max_temp: float, exponent_val: float) -> None:
    """
    @details Dynamic temperature implementation described in the paper https://arxiv.org/abs/2309.02772.

        LLAMA_API void llama_sample_entropy(
                struct llama_context * ctx,
              llama_token_data_array * candidates_p,
                               float   min_temp,
                               float   max_temp,
                               float   exponent_val);
    """
    ...
  def llama_sample_grammar(ctx: ffi.CData, candidates: ffi.CData, grammar: ffi.CData) -> None:
    """
    @details Apply constraints from grammar

        LLAMA_API void llama_sample_grammar(
                struct llama_context * ctx,
              llama_token_data_array * candidates,
          const struct llama_grammar * grammar);
    """
    ...
  def llama_sample_min_p(ctx: ffi.CData, candidates: ffi.CData, p: float, min_keep: int) -> None:
    """
    @details Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841

        LLAMA_API void llama_sample_min_p(
                struct llama_context * ctx,
              llama_token_data_array * candidates,
                               float   p,
                              size_t   min_keep);
    """
    ...
  def llama_sample_repetition_penalties(ctx: ffi.CData, candidates: ffi.CData, last_tokens: ffi.CData, penalty_last_n: int, penalty_repeat: float, penalty_freq: float, penalty_present: float) -> None:
    """
    @details Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
    @details Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.

        LLAMA_API void llama_sample_repetition_penalties(
                struct llama_context * ctx,
              llama_token_data_array * candidates,
                   const llama_token * last_tokens,
                              size_t   penalty_last_n,
                               float   penalty_repeat,
                               float   penalty_freq,
                               float   penalty_present);
    """
    ...
  def llama_sample_softmax(ctx: ffi.CData, candidates: ffi.CData) -> None:
    """
    @details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.

        LLAMA_API void llama_sample_softmax(
                struct llama_context * ctx,
              llama_token_data_array * candidates);
    """
    ...
  def llama_sample_tail_free(ctx: ffi.CData, candidates: ffi.CData, z: float, min_keep: int) -> None:
    """
    @details Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.

        LLAMA_API void llama_sample_tail_free(
                struct llama_context * ctx,
              llama_token_data_array * candidates,
                               float   z,
                              size_t   min_keep);
    """
    ...
  def llama_sample_temp(ctx: ffi.CData, candidates: ffi.CData, temp: float) -> None:
    """
        LLAMA_API void llama_sample_temp(
                struct llama_context * ctx,
              llama_token_data_array * candidates,
                               float   temp);
    """
    ...
  def llama_sample_token(ctx: ffi.CData, candidates: ffi.CData) -> ffi.CData:
    """
    @details Randomly selects a token from the candidates based on their probabilities using the RNG of ctx.

        LLAMA_API llama_token llama_sample_token(
                struct llama_context * ctx,
              llama_token_data_array * candidates);
    """
    ...
  def llama_sample_token_greedy(ctx: ffi.CData, candidates: ffi.CData) -> ffi.CData:
    """
    @details Selects the token with the highest probability.
    Does not compute the token probabilities. Use llama_sample_softmax() instead.

        LLAMA_API llama_token llama_sample_token_greedy(
                struct llama_context * ctx,
              llama_token_data_array * candidates);
    """
    ...
  def llama_sample_token_mirostat(ctx: ffi.CData, candidates: ffi.CData, tau: float, eta: float, m: int, mu: ffi.CData) -> ffi.CData:
    """
    @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.

        LLAMA_API llama_token llama_sample_token_mirostat(
                struct llama_context * ctx,
              llama_token_data_array * candidates,
                               float   tau,
                               float   eta,
                             int32_t   m,
                               float * mu);
    """
    ...
  def llama_sample_token_mirostat_v2(ctx: ffi.CData, candidates: ffi.CData, tau: float, eta: float, mu: ffi.CData) -> ffi.CData:
    """
    @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.

        LLAMA_API llama_token llama_sample_token_mirostat_v2(
                struct llama_context * ctx,
              llama_token_data_array * candidates,
                               float   tau,
                               float   eta,
                               float * mu);
    """
    ...
  def llama_sample_top_k(ctx: ffi.CData, candidates: ffi.CData, k: int, min_keep: int) -> None:
    """
    @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751

        LLAMA_API void llama_sample_top_k(
                struct llama_context * ctx,
              llama_token_data_array * candidates,
                             int32_t   k,
                              size_t   min_keep);
    """
    ...
  def llama_sample_top_p(ctx: ffi.CData, candidates: ffi.CData, p: float, min_keep: int) -> None:
    """
    @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751

        LLAMA_API void llama_sample_top_p(
                struct llama_context * ctx,
              llama_token_data_array * candidates,
                               float   p,
                              size_t   min_keep);
    """
    ...
  def llama_sample_typical(ctx: ffi.CData, candidates: ffi.CData, p: float, min_keep: int) -> None:
    """
    @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.

        LLAMA_API void llama_sample_typical(
                struct llama_context * ctx,
              llama_token_data_array * candidates,
                               float   p,
                              size_t   min_keep);
    """
    ...
  def llama_save_session_file(ctx: ffi.CData, path_session: ffi.CData, tokens: ffi.CData, n_token_count: int) -> bool:
    """
        LLAMA_API DEPRECATED(bool llama_save_session_file(
                struct llama_context * ctx,
                          const char * path_session,
                   const llama_token * tokens,
                              size_t   n_token_count),
            "use llama_state_save_file instead");
    """
    ...
  def llama_set_abort_callback(ctx: ffi.CData, abort_callback: ffi.CData, abort_callback_data: ffi.CData) -> None:
    """
    Set abort callback

        LLAMA_API void llama_set_abort_callback(struct llama_context * ctx, ggml_abort_callback abort_callback, void * abort_callback_data);
    """
    ...
  def llama_set_causal_attn(ctx: ffi.CData, causal_attn: bool) -> None:
    """
    Set whether to use causal attention or not
    If set to true, the model will only attend to the past tokens

        LLAMA_API void llama_set_causal_attn(struct llama_context * ctx, bool causal_attn);
    """
    ...
  def llama_set_n_threads(ctx: ffi.CData, n_threads: int, n_threads_batch: int) -> None:
    """
    Set the number of threads used for decoding
    n_threads is the number of threads used for generation (single token)
    n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)

        LLAMA_API void llama_set_n_threads(struct llama_context * ctx, uint32_t n_threads, uint32_t n_threads_batch);
    """
    ...
  def llama_set_rng_seed(ctx: ffi.CData, seed: int) -> None:
    """
    Sets the current rng seed.

        LLAMA_API void llama_set_rng_seed(struct llama_context * ctx, uint32_t seed);
    """
    ...
  def llama_set_state_data(ctx: ffi.CData, src: ffi.CData) -> int:
    """
        LLAMA_API DEPRECATED(size_t llama_set_state_data(
                struct llama_context * ctx,
                       const uint8_t * src),
            "use llama_state_set_data instead");
    """
    ...
  def llama_split_path(split_path: ffi.CData, maxlen: int, path_prefix: ffi.CData, split_no: int, split_count: int) -> int:
    """
    @details Build a split GGUF final path for this chunk.
    llama_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"
    Returns the split_path length.

        LLAMA_API int llama_split_path(char * split_path, size_t maxlen, const char * path_prefix, int split_no, int split_count);
    """
    ...
  def llama_split_prefix(split_prefix: ffi.CData, maxlen: int, split_path: ffi.CData, split_no: int, split_count: int) -> int:
    """
    @details Extract the path prefix from the split_path if and only if the split_no and split_count match.
    llama_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix = "/models/ggml-model-q4_0"
    Returns the split_prefix length.

        LLAMA_API int llama_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int split_no, int split_count);
    """
    ...
  def llama_state_get_data(ctx: ffi.CData, dst: ffi.CData) -> int:
    """
    Copies the state to the specified destination address.
    Destination needs to have allocated enough memory.
    Returns the number of bytes copied

        LLAMA_API size_t llama_state_get_data(
                struct llama_context * ctx,
                             uint8_t * dst);
    """
    ...
  def llama_state_get_size(ctx: ffi.CData) -> int:
    """
    Returns the maximum size in bytes of the state (rng, logits, embedding
    and kv_cache) - will often be smaller after compacting tokens

        LLAMA_API size_t llama_state_get_size(const struct llama_context * ctx);
    """
    ...
  def llama_state_load_file(ctx: ffi.CData, path_session: ffi.CData, tokens_out: ffi.CData, n_token_capacity: int, n_token_count_out: ffi.CData) -> bool:
    """
    Save/load session file

        LLAMA_API bool llama_state_load_file(
                struct llama_context * ctx,
                          const char * path_session,
                         llama_token * tokens_out,
                              size_t   n_token_capacity,
                              size_t * n_token_count_out);
    """
    ...
  def llama_state_save_file(ctx: ffi.CData, path_session: ffi.CData, tokens: ffi.CData, n_token_count: int) -> bool:
    """
        LLAMA_API bool llama_state_save_file(
                struct llama_context * ctx,
                          const char * path_session,
                   const llama_token * tokens,
                              size_t   n_token_count);
    """
    ...
  def llama_state_seq_get_data(ctx: ffi.CData, dst: ffi.CData, seq_id: ffi.CData) -> int:
    """
    Copy the KV cache of a single sequence into the specified buffer

        LLAMA_API size_t llama_state_seq_get_data(
                struct llama_context * ctx,
                             uint8_t * dst,
                        llama_seq_id   seq_id);
    """
    ...
  def llama_state_seq_get_size(ctx: ffi.CData, seq_id: ffi.CData) -> int:
    """
    Get the exact size needed to copy the KV cache of a single sequence

        LLAMA_API size_t llama_state_seq_get_size(
                struct llama_context * ctx,
                        llama_seq_id   seq_id);
    """
    ...
  def llama_state_seq_load_file(ctx: ffi.CData, filepath: ffi.CData, dest_seq_id: ffi.CData, tokens_out: ffi.CData, n_token_capacity: int, n_token_count_out: ffi.CData) -> int:
    """
        LLAMA_API size_t llama_state_seq_load_file(
                struct llama_context * ctx,
                          const char * filepath,
                        llama_seq_id   dest_seq_id,
                         llama_token * tokens_out,
                              size_t   n_token_capacity,
                              size_t * n_token_count_out);
    """
    ...
  def llama_state_seq_save_file(ctx: ffi.CData, filepath: ffi.CData, seq_id: ffi.CData, tokens: ffi.CData, n_token_count: int) -> int:
    """
        LLAMA_API size_t llama_state_seq_save_file(
                struct llama_context * ctx,
                          const char * filepath,
                        llama_seq_id   seq_id,
                   const llama_token * tokens,
                              size_t   n_token_count);
    """
    ...
  def llama_state_seq_set_data(ctx: ffi.CData, src: ffi.CData, dest_seq_id: ffi.CData) -> int:
    """
    Copy the sequence data (originally copied with `llama_state_seq_get_data`) into the specified sequence
    Returns:
    - Positive: Ok
    - Zero: Failed to load

        LLAMA_API size_t llama_state_seq_set_data(
                struct llama_context * ctx,
                       const uint8_t * src,
                        llama_seq_id   dest_seq_id);
    """
    ...
  def llama_state_set_data(ctx: ffi.CData, src: ffi.CData) -> int:
    """
    Set the state reading from the specified address
    Returns the number of bytes read

        LLAMA_API size_t llama_state_set_data(
                struct llama_context * ctx,
                       const uint8_t * src);
    """
    ...
  def llama_supports_gpu_offload() -> bool:
    """    LLAMA_API bool llama_supports_gpu_offload(void);"""
    ...
  def llama_supports_mlock() -> bool:
    """    LLAMA_API bool llama_supports_mlock      (void);"""
    ...
  def llama_supports_mmap() -> bool:
    """    LLAMA_API bool llama_supports_mmap       (void);"""
    ...
  def llama_synchronize(ctx: ffi.CData) -> None:
    """
    Wait until all computations are finished
    This is automatically done when using one of the functions below to obtain the computation results
    and is not necessary to call it explicitly in most cases

        LLAMA_API void llama_synchronize(struct llama_context * ctx);
    """
    ...
  def llama_time_us() -> int:
    """    LLAMA_API int64_t llama_time_us(void);"""
    ...
  def llama_token_bos(model: ffi.CData) -> ffi.CData:
    """
    Special tokens

        LLAMA_API llama_token llama_token_bos(const struct llama_model * model); // beginning-of-sentence
    """
    ...
  def llama_token_cls(model: ffi.CData) -> ffi.CData:
    """    LLAMA_API llama_token llama_token_cls(const struct llama_model * model); // classification"""
    ...
  def llama_token_eos(model: ffi.CData) -> ffi.CData:
    """    LLAMA_API llama_token llama_token_eos(const struct llama_model * model); // end-of-sentence"""
    ...
  def llama_token_eot(model: ffi.CData) -> ffi.CData:
    """    LLAMA_API llama_token llama_token_eot   (const struct llama_model * model); // End of infill middle"""
    ...
  def llama_token_get_score(model: ffi.CData, token: ffi.CData) -> float:
    """    LLAMA_API float llama_token_get_score(const struct llama_model * model, llama_token token);"""
    ...
  def llama_token_get_text(model: ffi.CData, token: ffi.CData) -> ffi.CData:
    """    LLAMA_API const char * llama_token_get_text(const struct llama_model * model, llama_token token);"""
    ...
  def llama_token_get_type(model: ffi.CData, token: ffi.CData) -> int:
    """    LLAMA_API enum llama_token_type llama_token_get_type(const struct llama_model * model, llama_token token);"""
    ...
  def llama_token_is_eog(model: ffi.CData, token: ffi.CData) -> bool:
    """
    Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)

        LLAMA_API bool llama_token_is_eog(const struct llama_model * model, llama_token token);
    """
    ...
  def llama_token_middle(model: ffi.CData) -> ffi.CData:
    """    LLAMA_API llama_token llama_token_middle(const struct llama_model * model); // Beginning of infill middle"""
    ...
  def llama_token_nl(model: ffi.CData) -> ffi.CData:
    """    LLAMA_API llama_token llama_token_nl (const struct llama_model * model); // next-line"""
    ...
  def llama_token_prefix(model: ffi.CData) -> ffi.CData:
    """
    Codellama infill tokens

        LLAMA_API llama_token llama_token_prefix(const struct llama_model * model); // Beginning of infill prefix
    """
    ...
  def llama_token_sep(model: ffi.CData) -> ffi.CData:
    """    LLAMA_API llama_token llama_token_sep(const struct llama_model * model); // sentence separator"""
    ...
  def llama_token_suffix(model: ffi.CData) -> ffi.CData:
    """    LLAMA_API llama_token llama_token_suffix(const struct llama_model * model); // Beginning of infill suffix"""
    ...
  def llama_token_to_piece(model: ffi.CData, token: ffi.CData, buf: ffi.CData, length: int, special: bool) -> int:
    """
    Token Id -> Piece.
    Uses the vocabulary in the provided context.
    Does not write null terminator to the buffer.
    User code is responsible to remove the leading whitespace of the first non-BOS token when decoding multiple tokens.
    @param special If true, special tokens are rendered in the output.

        LLAMA_API int32_t llama_token_to_piece(
                  const struct llama_model * model,
                               llama_token   token,
                                      char * buf,
                                   int32_t   length,
                                      bool   special);
    """
    ...
  def llama_tokenize(model: ffi.CData, text: ffi.CData, text_len: int, tokens: ffi.CData, n_tokens_max: int, add_special: bool, parse_special: bool) -> int:
    """
    @details Convert the provided text into tokens.
    @param tokens The tokens pointer must be large enough to hold the resulting tokens.
    @return Returns the number of tokens on success, no more than n_tokens_max
    @return Returns a negative number on failure - the number of tokens that would have been returned
    @param parse_special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
    as plaintext. Does not insert a leading space.

        LLAMA_API int32_t llama_tokenize(
            const struct llama_model * model,
                          const char * text,
                             int32_t   text_len,
                         llama_token * tokens,
                             int32_t   n_tokens_max,
                                bool   add_special,
                                bool   parse_special);
    """
    ...
  def llama_vocab_type(model: ffi.CData) -> int:
    """    LLAMA_API enum llama_vocab_type   llama_vocab_type  (const struct llama_model   * model);"""
    ...
  def open_memstream(__bufp: ffi.CData, __sizep: ffi.CData) -> ffi.CData:
    """FILE *open_memstream(char **__bufp, size_t *__sizep) __API_AVAILABLE(macos(10.13), ios(11.0), tvos(11.0), watchos(4.0));"""
    ...
  def pclose(arg: ffi.CData) -> int:
    """int	 pclose(FILE *) __swift_unavailable("Use posix_spawn APIs or NSTask instead. (On iOS, process spawning is unavailable.)");"""
    ...
  def perror(arg: ffi.CData) -> None:
    """void	 perror(const char *) __cold;"""
    ...
  def popen(arg: ffi.CData, arg2: ffi.CData) -> ffi.CData:
    """FILE	*popen(const char *, const char *) __DARWIN_ALIAS_STARTING(__MAC_10_6, __IPHONE_2_0, __DARWIN_ALIAS(popen)) __swift_unavailable("Use posix_spawn APIs or NSTask instead. (On iOS, process spawning is unavailable.)");"""
    ...
  def printf(arg: ffi.CData, *args2) -> int:
    """int	 printf(const char * __restrict, ...) __printflike(1, 2);"""
    ...
  def putc(arg: int, arg2: ffi.CData) -> int:
    """int	 putc(int, FILE *);"""
    ...
  def putc_unlocked(arg: int, arg2: ffi.CData) -> int:
    """int	 putc_unlocked(int, FILE *);"""
    ...
  def putchar(arg: int) -> int:
    """int	 putchar(int);"""
    ...
  def putchar_unlocked(arg: int) -> int:
    """int	 putchar_unlocked(int);"""
    ...
  def puts(arg: ffi.CData) -> int:
    """int	 puts(const char *);"""
    ...
  def putw(arg: int, arg2: ffi.CData) -> int:
    """int	 putw(int, FILE *);"""
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
  def remove(arg: ffi.CData) -> int:
    """int	 remove(const char *);"""
    ...
  def rename(__old: ffi.CData, __new: ffi.CData) -> int:
    """int	 rename (const char *__old, const char *__new);"""
    ...
  def renameat(arg: int, arg2: ffi.CData, arg3: int, arg4: ffi.CData) -> int:
    """int     renameat(int, const char *, int, const char *) __OSX_AVAILABLE_STARTING(__MAC_10_10, __IPHONE_8_0);"""
    ...
  def renameatx_np(arg: int, arg2: ffi.CData, arg3: int, arg4: ffi.CData, arg5: int) -> int:
    """int renameatx_np(int, const char *, int, const char *, unsigned int) __OSX_AVAILABLE(10.12) __IOS_AVAILABLE(10.0) __TVOS_AVAILABLE(10.0) __WATCHOS_AVAILABLE(3.0);"""
    ...
  def renamex_np(arg: ffi.CData, arg2: ffi.CData, arg3: int) -> int:
    """int renamex_np(const char *, const char *, unsigned int) __OSX_AVAILABLE(10.12) __IOS_AVAILABLE(10.0) __TVOS_AVAILABLE(10.0) __WATCHOS_AVAILABLE(3.0);"""
    ...
  def rewind(arg: ffi.CData) -> None:
    """void	 rewind(FILE *);"""
    ...
  def scanf(arg: ffi.CData, *args2) -> int:
    """int	 scanf(const char * __restrict, ...) __scanflike(1, 2);"""
    ...
  def setbuf(arg: ffi.CData, arg2: ffi.CData) -> None:
    """void	 setbuf(FILE * __restrict, char * __restrict);"""
    ...
  def setbuffer(arg: ffi.CData, arg2: ffi.CData, arg3: int) -> None:
    """void	 setbuffer(FILE *, char *, int);"""
    ...
  def setlinebuf(arg: ffi.CData) -> int:
    """int	 setlinebuf(FILE *);"""
    ...
  def setvbuf(arg: ffi.CData, arg2: ffi.CData, arg3: int, arg4: int) -> int:
    """int	 setvbuf(FILE * __restrict, char * __restrict, int, size_t);"""
    ...
  def snprintf(__str: ffi.CData, __size: int, __format: ffi.CData, *args2) -> int:
    """int	 snprintf(char * __restrict __str, size_t __size, const char * __restrict __format, ...) __printflike(3, 4);"""
    ...
  def sprintf(arg: ffi.CData, arg2: ffi.CData, *args2) -> int:
    """int	 sprintf(char * __restrict, const char * __restrict, ...) __printflike(2, 3);"""
    ...
  def sscanf(arg: ffi.CData, arg2: ffi.CData, *args2) -> int:
    """int	 sscanf(const char * __restrict, const char * __restrict, ...) __scanflike(2, 3);"""
    ...
  def start_rpc_server(backend: ffi.CData, endpoint: ffi.CData, free_mem: int, total_mem: int) -> None:
    """GGML_API GGML_CALL void start_rpc_server(ggml_backend_t backend, const char * endpoint, size_t free_mem, size_t total_mem);"""
    ...
  def tempnam(__dir: ffi.CData, __prefix: ffi.CData) -> ffi.CData:
    """char	*tempnam(const char *__dir, const char *__prefix) __DARWIN_ALIAS(tempnam);"""
    ...
  def tmpfile() -> ffi.CData:
    """FILE	*tmpfile(void);"""
    ...
  def tmpnam(arg: ffi.CData) -> ffi.CData:
    """char	*tmpnam(char *);"""
    ...
  def ungetc(arg: int, arg2: ffi.CData) -> int:
    """int	 ungetc(int, FILE *);"""
    ...
  def vasprintf(arg: ffi.CData, arg2: ffi.CData, arg3: ffi.CData) -> int:
    """int	 vasprintf(char ** __restrict, const char * __restrict, va_list) __printflike(2, 0);"""
    ...
  def vdprintf(arg: int, arg2: ffi.CData, arg3: ffi.CData) -> int:
    """int	vdprintf(int, const char * __restrict, va_list) __printflike(2, 0) __OSX_AVAILABLE_STARTING(__MAC_10_7, __IPHONE_4_3);"""
    ...
  def vfprintf(arg: ffi.CData, arg2: ffi.CData, arg3: ffi.CData) -> int:
    """int	 vfprintf(FILE * __restrict, const char * __restrict, va_list) __printflike(2, 0);"""
    ...
  def vfscanf(__stream: ffi.CData, __format: ffi.CData, arg: ffi.CData) -> int:
    """int	 vfscanf(FILE * __restrict __stream, const char * __restrict __format, va_list) __scanflike(2, 0);"""
    ...
  def vprintf(arg: ffi.CData, arg2: ffi.CData) -> int:
    """int	 vprintf(const char * __restrict, va_list) __printflike(1, 0);"""
    ...
  def vscanf(__format: ffi.CData, arg: ffi.CData) -> int:
    """int	 vscanf(const char * __restrict __format, va_list) __scanflike(1, 0);"""
    ...
  def vsnprintf(__str: ffi.CData, __size: int, __format: ffi.CData, arg: ffi.CData) -> int:
    """int	 vsnprintf(char * __restrict __str, size_t __size, const char * __restrict __format, va_list) __printflike(3, 0);"""
    ...
  def vsprintf(arg: ffi.CData, arg2: ffi.CData, arg3: ffi.CData) -> int:
    """int	 vsprintf(char * __restrict, const char * __restrict, va_list) __printflike(2, 0);"""
    ...
  def vsscanf(__str: ffi.CData, __format: ffi.CData, arg: ffi.CData) -> int:
    """int	 vsscanf(const char * __restrict __str, const char * __restrict __format, va_list) __scanflike(2, 0);"""
    ...