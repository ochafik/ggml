# Ported from llama2.c
# https://github.com/karpathy/llama2.c/blob/master/run.c
# Use faster tokenization, and naively allocate buffers in GGML / wrap them to numpy

# rm -fR /tmp/llama_release ; cmake ../../../llama.cpp -B /tmp/llama_release -DCMAKE_C_FLAGS=-Ofast -DLLAMA_NATIVE=1 -DLLAMA_LTO=1 -DLLAMA_METAL=1 -DBUILD_SHARED_LIBS=1 -DLLAMA_BUILD_EXAMPLES=0 -DLLAMA_BUILD_TESTS=0 -DCMAKE_BUILD_TYPE=Release && ( cd /tmp/llama_release && make -j ) && cp ../../../llama.cpp/ggml-metal.metal /tmp/llama_release/
import os; os.environ['GGML_LIBRARY'] = '/tmp/llama_release/libggml_shared.dylib'

# rm -fR /tmp/llama_debug ; cmake ../../../llama.cpp -B /tmp/llama_debug -DLLAMA_METAL=1 -DBUILD_SHARED_LIBS=1 -DLLAMA_BUILD_EXAMPLES=0 -DLLAMA_BUILD_TESTS=0 -DCMAKE_BUILD_TYPE=Debug && ( cd /tmp/llama_debug && make -j ) && cp ../../../llama.cpp/ggml-metal.metal /tmp/llama_debug/
# import os; os.environ['GGML_LIBRARY'] = '/tmp/llama_debug/libggml_shared.dylib'

# python llama2.c.py ~/AI/Models/llama2.c.stories15M.bin ../../../llama2.c/tokenizer.bin --prompt "Hello, world"

from ggml import lib, ffi
from ggml.utils import init, numpy, copy, debug_ggml_asserts, debug_str
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict, Any, Callable, TypeVar, Generic
from jsonargparse import CLI
from pathlib import Path
import struct, os, mmap, math, sys, time, bisect
from collections import namedtuple
import scipy as sp

MMAP = os.environ.get('MMAP', '0') == '1'
ALLOC = os.environ.get('ALLOC', '0') == '1'
METAL = os.environ.get('METAL', '0') == '1'

Tensor = ffi.CData
TensorLike = Union[Tensor, np.ndarray]

GGML_MAX_NODES = 4096

_eternal_cstrs = {}
def _const_cstr(s: str):
    cstr = _eternal_cstrs.get(s)
    if cstr is None:
        cstr = ffi.new('char[]', s)
        _eternal_cstrs[s] = cstr
    return cstr

def _get_shape(x: TensorLike): return x.shape if isinstance(x, np.ndarray) else tuple([x.ne[i] for i in range(x.n_dims)])

class AlignedBuf:
    def __init__(self, size: int, alignment=4096):
        self.size = size
        self.raw_ptr = ffi.new('char[]', size + alignment - 1)
        addr = int(ffi.cast('size_t', self.raw_ptr))
        self.ptr = ffi.cast('char*', addr + (alignment - (addr % alignment)))

# ----------------------------------------------------------------------------
# Transformer, KVCache and Vocabulary structs, and related memory management

@dataclass
class Config:
    dim: int # transformer dimension
    hidden_dim: int # for ffn layers
    n_layers: int # number of layers
    n_heads: int # number of query heads
    n_kv_heads: int # number of key/value heads (can be < query heads because of multiquery)
    vocab_size: int # vocabulary size, usually 256 (byte-level)
    seq_len: int # max sequence length

    @property
    def head_size(self): return self.dim // self.n_heads
    @property
    def n_embd_gqa(self): return self.dim // self.n_gqa
    @property
    def n_gqa(self): return self.n_heads // self.n_kv_heads

@dataclass
class LlamaLayer:
    attn_norm: Tensor = None
    attn_norm_b: Optional[Tensor] = None
    attn_norm_2: Optional[Tensor] = None
    attn_norm_2_b: Optional[Tensor] = None

    # attention
    wq: Optional[Tensor] = None
    wk: Optional[Tensor] = None
    wv: Optional[Tensor] = None
    wo: Tensor = None
    wqkv: Optional[Tensor] = None

    # normalization
    ffn_norm: Tensor = None

    # ff
    w1: Tensor = None # ffn_gat
    w2: Tensor = None # ffn_dow
    w3: Tensor = None # ffn_u

@dataclass
class LlamaModel:
    ctx: ffi.CData
    buf: AlignedBuf = None

    tok_embeddings: Tensor = None
    output_norm: Tensor = None
    output_norm_b: Tensor = None
    output: Tensor = None
    layers: List[LlamaLayer] = None

# struct used when sorting probabilities during top-p sampling
@dataclass
class ProbIndex:
    prob: float
    index: int

@dataclass
class KVCache:
    def __init__(self, p: Config, type: int):
        self.buf = AlignedBuf(
                    2 * ( \
                        lib.ggml_tensor_overhead() + \
                        p.n_layers * p.seq_len * p.n_embd_gqa * lib.ggml_type_size(type)))
        params = ffi.new('struct ggml_init_params*')
        params.mem_size = self.buf.size
        params.mem_buffer = self.buf.ptr
        params.no_alloc = False
        self.ctx = lib.ggml_init(params[0])
        assert self.ctx

        self.k = lib.ggml_new_tensor_3d(self.ctx, type, p.n_layers, p.seq_len, p.n_embd_gqa)
        self.v = lib.ggml_new_tensor_3d(self.ctx, type, p.n_layers, p.seq_len, p.n_embd_gqa)
        lib.ggml_set_name(self.k, _const_cstr(b"cache_k\0"))
        lib.ggml_set_name(self.v, _const_cstr(b"cache_v\0"))
        self.cache_length = 0

@dataclass
class TokenIndex:
    str: str
    id: int

@dataclass
class Vocabulary:
    vocab_size: int 
    vocab: list[str]
    vocab_scores: list[float]
    max_token_length: int
    # Sorted vocabulary tokens and their corresponding id for log(vocab_size) lookups.
    sorted_vocab: list[TokenIndex]

rms_norm_eps = 1e-5
rope_freq_base  = 10000.0
rope_freq_scale = 1.0

@dataclass
class EvalParams:
    n_threads: int # number of threads to use
    # n_tokens: int # number of tokens
    n_past: int # the context size so far
    
    tokens: Optional[np.ndarray] = None # new batch of tokens to process
    embd: Optional[np.ndarray] = None # embeddings input

    # TODO: options:
    # - No output (just kv-cache building) if Temperature = None
    # - Softmax(Logits / Temperature) if Temperature != 0
    # - Argmax(Logits) if Temperature == 0
    # - ?? Raw embeddings
    # - ? Raw logits (useful for distillation?)
    # - Other sampling methods? Mirostat, etc

    all_logits: bool = False
    # extract_embeddings: bool = True
    temperature: Optional[float] = None

    @property
    def n_tokens(self):
        if self.tokens is not None:
            (N,) = self.tokens.shape
            return N
        else:
            (dim, N) = self.embd.shape
            assert dim == self.n_embd
            return N

@dataclass
class GraphResult:
    graph: ffi.CData
    # embeddings: Optional[Tensor] = None
    logits: Optional[Tensor] = None
    argmax: Optional[Tensor] = None

class LlamaContext:
    def __init__(self, p: Config, kv_cache: KVCache, model: LlamaModel):
        self.p = p
        self.kv_self = kv_cache
        self.model = model
    
        self.work_buffer = ffi.NULL

        if METAL:
            self.ctx_metal = lib.ggml_metal_init(1)
            assert self.ctx_metal, "ggml_metal_init() failed"

        if ALLOC:
            tensor_alignment = 32
            # the compute buffer is used to store the tensor and graph structs, while the allocator buffer is used for the tensor data
            self.buf_compute = AlignedBuf(lib.ggml_tensor_overhead() * GGML_MAX_NODES + lib.ggml_graph_overhead())

            # create measure allocator
            self.alloc = lib.ggml_allocr_new_measure(tensor_alignment)

            # build worst-case graph
            result = self.llama_build_graph(EvalParams(
                all_logits=False, temperature=1.0,
                n_threads=1, n_past=0,
                tokens=np.array([llama_token_bos() for _ in range(p.seq_len)], dtype=np.int32)))
            
            if METAL and self.ctx_metal:
                lib.ggml_metal_graph_find_concurrency(self.ctx_metal, result.graph, False)
                self.allocr_set_metal_parse_seq()

            # measure memory requirements for the graph
            alloc_size = lib.ggml_allocr_alloc_graph(self.alloc, result.graph) + tensor_alignment
            print(f'alloc_size: {alloc_size}')
            print(f'buf_compute.size: {self.buf_compute.size}')

            # recreate allocator with exact memory requirements
            lib.ggml_allocr_free(self.alloc)

            self.buf_alloc = AlignedBuf(alloc_size)
            self.alloc = lib.ggml_allocr_new(self.buf_alloc.ptr, alloc_size, tensor_alignment)
            
            if METAL and self.ctx_metal:
                self.allocr_set_metal_parse_seq()
        else:
            self.alloc = None

            self.scratch_size = 500*1024*1024
            # self.scratch_size = 1024*1024*1024
            # We have 1 more scratch buffer than traditional llama.cpp because of the early exit in the last layer
            self.buf_scratch = [AlignedBuf(self.scratch_size) for _ in range(3)]
            self.buf_scratch_max_size = [0 for _ in self.buf_scratch]
            self.buf_last = 0

            # ffi.new('char[]', 500*1024*1024)
            # self.work_buffer = ffi.new('char[]', 1024*1024*1024)
            self.tmp_ggml_scratch = ffi.new('struct ggml_scratch*')

        if METAL and self.ctx_metal:
            if MMAP:
                self.add_buf_to_metal("data", model.mapping)
            else:
                self.add_ctx_to_metal("data", model.ctx, use_max_tensor_size=True)
            self.add_buf_to_metal("eval", self.buf_compute)
            self.add_buf_to_metal("kv", self.kv_self.buf)
            self.add_buf_to_metal("alloc", self.buf_alloc)
            # self.add_buf_to_metal("work", self.work_buffer)

    def allocr_set_metal_parse_seq(self):
        lib.ggml_allocr_set_parse_seq(self.alloc, lib.ggml_metal_get_concur_list(self.ctx_metal), lib.ggml_metal_if_optimized(self.ctx_metal))
                
    def add_buf_to_metal(self, name: str, buf: AlignedBuf):
        lib.ggml_metal_add_buffer(self.ctx_metal,
                                  _const_cstr(name.encode()),
                                  buf.ptr, buf.size, 0)

    def add_ctx_to_metal(self, name: str, ctx, use_max_tensor_size=False):
        mem_buffer = lib.ggml_get_mem_buffer(ctx)
        mem_size = lib.ggml_get_mem_size(ctx)
        max_tensor_size = lib.ggml_get_max_tensor_size(ctx)
        assert mem_buffer != ffi.NULL and mem_size > 0
        lib.ggml_metal_add_buffer(self.ctx_metal, _const_cstr(name.encode()),
                                    mem_buffer,
                                    mem_size,
                                    max_tensor_size if use_max_tensor_size else 0)

    def ggml_graph_compute_helper(self, graph, n_threads: int):
        # print(f'n_threads: {n_threads}')
        plan = lib.ggml_graph_plan(graph, n_threads)

        if plan.work_size > 0:
            # assert ffi.sizeof(self.work_buffer) >= plan.work_size, f'work_size: {plan.work_size} > {ffi.sizeof(self.work_buffer)}'
            if not self.work_buffer or plan.work_size > ffi.sizeof(self.work_buffer):
                self.work_buffer = ffi.new('char[]', plan.work_size)
            plan.work_data = self.work_buffer

        lib.ggml_graph_compute(graph, ffi.addressof(plan))

    def use_buf(self, ctx, i):
        if self.alloc:
            return
        
        scratch = self.tmp_ggml_scratch
        last_size = 0

        if i == -1:
            scratch.offs = 0
            scratch.size = 0
            scratch.data = ffi.NULL
            lib.ggml_set_scratch(ctx, scratch[0])
        else:
            scratch.offs = 0
            scratch.size = self.scratch_size
            scratch.data = self.buf_scratch[i].ptr
            last_size = lib.ggml_set_scratch(ctx, scratch[0])   
        
        if self.buf_last >= 0:
            self.buf_scratch_max_size[self.buf_last] = max(self.buf_scratch_max_size[self.buf_last], last_size)

        self.buf_last = i
     
    def get_buf_max_mem(self, i): return self.buf_scratch_max_size[i]
    
    def llama_eval(self, ep: EvalParams) -> GraphResult:
        N = ep.n_tokens

        if self.alloc: lib.ggml_allocr_reset(self.alloc)
        result = self.llama_build_graph(ep)
        if self.alloc: lib.ggml_allocr_alloc_graph(self.alloc, result.graph)

        # for big prompts, if BLAS is enabled, it is better to use only one thread
        # otherwise, the threads are spin-lock waiting for the BLAS calls and are degrading the performance
        if N >= 32 and lib.ggml_cpu_has_blas() and not lib.ggml_cpu_has_gpublas():
            n_threads = 1
        else:
            n_threads = ep.n_threads
        # n_threads = ep.n_threads
        
        if METAL and self.ctx_metal:
            lib.ggml_metal_set_n_cb     (self.ctx_metal, n_threads)
            lib.ggml_metal_graph_compute(self.ctx_metal, result.graph)
            lib.ggml_metal_get_tensor   (self.ctx_metal, result.logits)
            lib.ggml_metal_get_tensor   (self.ctx_metal, result.logits or result.argmax)
            # if (!lctx.embedding.empty()) {
            #     ggml_metal_get_tensor(lctx.ctx_metal, embeddings);
            # }
        else:
            self.ggml_graph_compute_helper(result.graph, n_threads)

        # update kv token count
        self.kv_self.cache_length = ep.n_past + N

        return result

    def llama_build_graph(self, ep: EvalParams) -> GraphResult:
        assert ((ep.tokens is None and ep.embd) or (ep.tokens is not None and not ep.embd))

        p = self.p
        model = self.model

        N = ep.n_tokens
        n_past = ep.n_past

        n_gqa = p.n_gqa
        n_embd = p.dim
        n_embd_gqa = p.n_embd_gqa
        n_embd_head = p.head_size
        n_layer = p.n_layers
        n_ctx = p.seq_len
        n_head = p.n_heads
        n_head_kv = p.n_kv_heads
        
        params = ffi.new('struct ggml_init_params*')
        if self.alloc:
            params.mem_size = self.buf_compute.size
            params.mem_buffer = self.buf_compute.ptr
            params.no_alloc = True
        else:
            params.mem_size = 500*1024*1024
            # params.mem_size = 1024*1024*1024
            params.mem_buffer = ffi.NULL
            params.no_alloc = False
        ctx0 = lib.ggml_init(params[0])

        def prepare_copy(tensor):
            if not self.alloc:
                return True
            lib.ggml_allocr_alloc(self.alloc, tensor)
            return not lib.ggml_allocr_is_measure(self.alloc)

        try:
            gf = lib.ggml_new_graph(ctx0)

            if ep.tokens is not None:
                inp_tokens = lib.ggml_new_tensor_1d(ctx0, lib.GGML_TYPE_I32, N)

                if prepare_copy(inp_tokens):
                    copy(ep.tokens, inp_tokens)
                    # ffi.memmove(inp_tokens.data, np.array(ep.tokens, dtype=int), N*lib.ggml_element_size(inp_tokens))

                inpL = lib.ggml_get_rows(ctx0, model.tok_embeddings, inp_tokens)
            else:
                inpL = lib.ggml_new_tensor_2d(ctx0, lib.GGML_TYPE_F32, n_embd, N)
                if prepare_copy(inpL):
                    copy(np.array(ep.embd, dtype=float), inpL)
                    # ffi.memmove(inpL.data, ffi.frombuffer(ep.embd), N * n_embd * lib.ggml_element_size(inpL))

            assert _get_shape(inpL) == (n_embd, N), f'Bad input shape: {_get_shape(inpL)} != {(n_embd, N)}'

            KQ_scale = lib.ggml_new_tensor_1d(ctx0, lib.GGML_TYPE_F32, 1)
            if prepare_copy(KQ_scale):
                lib.ggml_set_f32(KQ_scale, 1.0/math.sqrt(float(n_embd)/n_head))

            for il in range(n_layer):
                inpSA = inpL

                focus_on_last_token_after_kv_cache_write = il == n_layer - 1 and not ep.all_logits
                # focus_on_last_token_after_kv_cache_write = False
                self.use_buf(ctx0, 0)
                # self.use_buf(ctx0, 2 if focus_on_last_token_after_kv_cache_write else 0)

                #  norm
                cur = lib.ggml_rms_norm(ctx0, inpL, rms_norm_eps)
                #  cur = cur*attention_norm(broadcasted)
                cur = lib.ggml_mul(ctx0, cur, model.layers[il].attn_norm)

                #  self-attention
                
                #  compute Q and K and RoPE them
                tmpk = lib.ggml_mul_mat(ctx0, model.layers[il].wk, cur)
                Kcur = lib.ggml_rope_custom_inplace(
                    ctx0, lib.ggml_reshape_3d(ctx0, tmpk, n_embd_head, n_head_kv, N),
                    n_past, n_embd_head, 0, 0,
                    rope_freq_base, rope_freq_scale)

                #  store key and value to memory
                
                #  compute the transposed [N, n_embd] V matrix
                tmpv = lib.ggml_mul_mat(ctx0, model.layers[il].wv, cur)
                Vcur = lib.ggml_transpose(ctx0, lib.ggml_reshape_2d(ctx0, tmpv, n_embd_gqa, N))
                
                k = lib.ggml_view_1d(ctx0, self.kv_self.k, N*n_embd_gqa,
                                    (lib.ggml_element_size(self.kv_self.k)*n_embd_gqa)*(il*n_ctx + n_past))
                v = lib.ggml_view_2d(ctx0, self.kv_self.v, N, n_embd_gqa,
                        (   n_ctx)*lib.ggml_element_size(self.kv_self.v),
                        (il*n_ctx)*lib.ggml_element_size(self.kv_self.v)*n_embd_gqa + n_past*lib.ggml_element_size(self.kv_self.v))
                

                #  important: storing RoPE-ed version of K in the KV cache!
                lib.ggml_build_forward_expand(gf, lib.ggml_cpy(ctx0, Kcur, k))
                lib.ggml_build_forward_expand(gf, lib.ggml_cpy(ctx0, Vcur, v))

                if focus_on_last_token_after_kv_cache_write:
                # if il == n_layer - 1 and not ep.all_logits:
                    # From here on, we only care about the last token and its logits.
                    # We do as if N = 1 (from the end)
                    # This means we only keep the last chunk of cur and inpSA
                    assert   cur.n_dims == 2 and   cur.ne[0] == n_embd and   cur.ne[1] == N
                    assert inpSA.n_dims == 2 and inpSA.ne[0] == n_embd and inpSA.ne[1] == N
                    # assert cur.nb[1] == N*lib.ggml_element_size(cur), f'Bad cur nb: {cur.nb[1]} != {N*lib.ggml_element_size(cur)}'
                    cur   = lib.ggml_view_2d(ctx0, cur,   n_embd, 1,   cur.nb[1], (N-1)*lib.ggml_element_size(cur) * n_embd)
                    inpSA = lib.ggml_view_2d(ctx0, inpSA, n_embd, 1, inpSA.nb[1], (N-1)*lib.ggml_element_size(inpSA) * n_embd)

                    # Make cur and inpSA contiguous in memory to speed up the matmul, however we waste time on the copy
                    # cur   = lib.ggml_cpy(ctx0, cur,   lib.ggml_new_tensor_2d(ctx0, lib.GGML_TYPE_F32, n_embd, 1))
                    # inpSA = lib.ggml_cpy(ctx0, inpSA, lib.ggml_new_tensor_2d(ctx0, lib.GGML_TYPE_F32, n_embd, 1))
                    
                    n_past += N - 1
                    N = 1

                tmpq = lib.ggml_mul_mat(ctx0, model.layers[il].wq, cur)
                Qcur = lib.ggml_rope_custom_inplace(
                    ctx0, lib.ggml_reshape_3d(ctx0, tmpq, n_embd_head, n_head, N),
                    n_past, n_embd_head, 0, 0,
                    rope_freq_base, rope_freq_scale)

                Q = lib.ggml_permute(ctx0, Qcur, 0, 2, 1, 3)
                K = lib.ggml_permute(ctx0, lib.ggml_reshape_3d(ctx0,
                                lib.ggml_view_1d(ctx0, self.kv_self.k, (n_past + N)*n_embd_gqa, il*n_ctx*lib.ggml_element_size(self.kv_self.k)*n_embd_gqa),
                                n_embd_head, n_head_kv, n_past + N),
                            0, 2, 1, 3)
                #  K * Q
                KQ = lib.ggml_mul_mat(ctx0, K, Q)
                #  KQ_scaled = KQ / sqrt(n_embd_head)
                #  KQ_scaled shape [n_past + N, N, n_head, 1]
                KQ_scaled = lib.ggml_scale_inplace(ctx0, KQ, KQ_scale)
                #  KQ_masked = mask_past(KQ_scaled)
                KQ_masked = lib.ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past)
                #  KQ = soft_max(KQ_masked)
                KQ_soft_max = lib.ggml_soft_max_inplace(ctx0, KQ_masked)
                #  split cached V into n_head heads
                V = lib.ggml_view_3d(ctx0, self.kv_self.v,
                            n_past + N, n_embd_head, n_head_kv,
                            n_ctx*lib.ggml_element_size(self.kv_self.v),
                            n_ctx*lib.ggml_element_size(self.kv_self.v)*n_embd_head,
                            n_ctx*lib.ggml_element_size(self.kv_self.v)*n_embd_gqa*il)

                if True:
                    KQV = lib.ggml_mul_mat(ctx0, V, KQ_soft_max)
                else:
                    #  make V contiguous in memory to speed up the matmul, however we waste time on the copy
                    #  on M1 this is faster for the perplexity computation, but ~5% slower for the single-token generation
                    #  is there a better way?
                    V_cont = lib.ggml_cpy(ctx0, V, lib.ggml_new_tensor_3d(ctx0, self.kv_self.v.type, n_past + N, n_embd_head, n_head))
                    KQV = lib.ggml_mul_mat(ctx0, V_cont, KQ_soft_max)
                #  KQV_merged = KQV.permute(0, 2, 1, 3)
                KQV_merged = lib.ggml_permute(ctx0, KQV, 0, 2, 1, 3)
                #  cur = KQV_merged.contiguous().view(n_embd, N)
                cur = lib.ggml_cpy(ctx0,
                        KQV_merged,
                        lib.ggml_new_tensor_2d(ctx0, lib.GGML_TYPE_F32, n_embd, N))
                #  projection (no bias)
                cur = lib.ggml_mul_mat(ctx0, model.layers[il].wo, cur)
                
                self.use_buf(ctx0, 1)

                inpFF = lib.ggml_add(ctx0, cur, inpSA)

                #  feed-forward network
                
                #  norm
                cur = lib.ggml_rms_norm(ctx0, inpFF, rms_norm_eps)
                #  cur = cur*ffn_norm(broadcasted)
                cur = lib.ggml_mul(ctx0, cur, model.layers[il].ffn_norm)
                tmp = lib.ggml_mul_mat(ctx0, model.layers[il].w3, cur)
                cur = lib.ggml_mul_mat(ctx0, model.layers[il].w1, cur)
                #  SILU activation
                cur = lib.ggml_silu(ctx0, cur)
                cur = lib.ggml_mul(ctx0, cur, tmp)
                cur = lib.ggml_mul_mat(ctx0, model.layers[il].w2, cur)
                cur = lib.ggml_add(ctx0, cur, inpFF)
                #  input for next layer
                inpL = cur

            self.use_buf(ctx0, 0)

            #  norm
            cur = lib.ggml_rms_norm(ctx0, inpL, rms_norm_eps)

            #  cur = cur*norm(broadcasted)
            cur = lib.ggml_mul(ctx0, cur, model.output_norm)
            lib.ggml_set_name(cur, _const_cstr(b"result_norm\0"))

            #  lm_head
            cur = lib.ggml_mul_mat(ctx0, model.output, cur)
            lib.ggml_set_name(cur, _const_cstr(b"result_output\0"))
            
            self.use_buf(ctx0, -1)

            #  logits -> probs
            # cur = lib.ggml_soft_max_inplace(ctx0, cur)

            if ep.temperature is not None:
                if ep.temperature == 0.0:
                    # print('argmax')
                    cur = lib.ggml_argmax(ctx0, cur)
                else:
                    cur = lib.ggml_scale_inplace(ctx0, cur, lib.ggml_new_f32(ctx0, 1.0/ep.temperature))
                    cur = lib.ggml_soft_max_inplace(ctx0, cur)
            
            lib.ggml_build_forward_expand(gf, cur)

            res = gf.nodes[gf.n_nodes - 1]
            if ep.temperature == 0.0:
                assert _get_shape(res) == (N, 1), f'Bad argmax shape: {_get_shape(res)} != {(N, 1)}'
            else:
                assert _get_shape(res) == (p.vocab_size, N), f'Bad output shape: {_get_shape(res)} != {(p.vocab_size, N)}'
            # assert ffi.string(res.name).decode('utf-8').startswith("result_output"), ffi.string(res.name)
            
            # return GraphResult(graph=gf, logits=res)
            return GraphResult(graph=gf, argmax=res) if ep.temperature == 0.0 \
                else GraphResult(graph=gf, logits=res)

        finally:
            lib.ggml_free(ctx0)

def lookup_token(str: str, v: Vocabulary):
    t = v.sorted_vocab[bisect.bisect_left(v.sorted_vocab, str, key=lambda x: x.str)]
    return t.id if t.str == str else -1

def lookup_merge(token1: int, token2: int, v: Vocabulary) -> int:
    return lookup_token(f"{v.vocab[token1]}{v.vocab[token2]}", v)

def bpe_encode(text: str, v: Vocabulary) -> list[int]:
    tokens = []

    # first encode every individual byte in the input string
    tokens = [lookup_token(c, v) for c in text]
    assert [x for x in tokens if x == -1] == []
    
    possible_merges = [lookup_merge(tokens[i], tokens[i+1], v) for i in range(len(tokens)-1)]

    # merge the best consecutive pair each iteration, according the scores in vocab_scores
    while True:
        best_score = -1e10
        best_id = -1
        best_idx = -1

        for i in range(len(tokens)-1):
            # check if we can merge the pair (tokens[i], tokens[i+1])
            id = possible_merges[i]
            if id != -1 and v.vocab_scores[id] > best_score:
                # this merge pair exists in vocab! record its score and position
                best_score = v.vocab_scores[id]
                best_id = id
                best_idx = i

        if best_idx == -1:
            break # we couldn't find any more pairs to merge, so we're done
        
        # merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens.pop(best_idx)
        possible_merges.pop(best_idx)
        tokens[best_idx] = best_id

        # update possible merge pairs on the left and right of the merged pair
        if best_idx < len(tokens)-1:
            possible_merges[best_idx] = lookup_merge(tokens[best_idx],   tokens[best_idx+1], v)
        if best_idx > 0:
            possible_merges[best_idx-1] = lookup_merge(tokens[best_idx-1], tokens[best_idx], v)

    return tokens

def new_tensor(ctx, ttype, *shape: list[int]):
    if len(shape) == 1:
        return lib.ggml_new_tensor_1d(ctx, ttype, shape[0])
    elif len(shape) == 2:
        return lib.ggml_new_tensor_2d(ctx, ttype, shape[0], shape[1])
    elif len(shape) == 3:
        return lib.ggml_new_tensor_3d(ctx, ttype, shape[0], shape[1], shape[2])
    elif len(shape) == 4:
        return lib.ggml_new_tensor_4d(ctx, ttype, shape[0], shape[1], shape[2], shape[3])
    else:
        raise Exception(f'Bad shape: {shape}')

def read_llama2c_model(mm, p: Config, shared_weights: bool) -> LlamaModel:
    model = LlamaModel(ctx = init(mem_size=30*1024*1024*1024))
    model.layers = [LlamaLayer() for _ in range(p.n_layers)]

    def read_tensor(name: str, *shape: Tuple[int, ...]):
        nbytes = np.prod(shape) * ffi.sizeof('float')
        # TODO: make type depend on name as in quantization routine
        ttype = lib.GGML_TYPE_F32
        array = np.frombuffer(mm.read(nbytes), dtype=np.float32).reshape(shape)
        tensor = new_tensor(model.ctx, ttype, *shape)
        lib.ggml_set_name(tensor, _const_cstr(name.encode()))
        
        copy(array, tensor)
        return tensor
        
    model.tok_embeddings = read_tensor("token_embd", p.dim, p.vocab_size)
    for i in range(p.n_layers): model.layers[i].attn_norm = read_tensor(f'blk.{i}.attn_norm', p.dim)
    for i in range(p.n_layers): model.layers[i].wq = read_tensor(f'blk.{i}.attn_q', p.dim, p.dim)
    for i in range(p.n_layers): model.layers[i].wk = read_tensor(f'blk.{i}.attn_k', p.dim, p.dim)
    for i in range(p.n_layers): model.layers[i].wv = read_tensor(f'blk.{i}.attn_v', p.dim, p.dim)
    for i in range(p.n_layers): model.layers[i].wo = read_tensor(f'blk.{i}.attn_output', p.dim, p.dim)
    for i in range(p.n_layers): model.layers[i].ffn_norm = read_tensor(f'blk.{i}.ffn_norm', p.dim)
    for i in range(p.n_layers): model.layers[i].w1 = read_tensor(f'blk.{i}.ffn_gate', p.dim, p.hidden_dim)
    for i in range(p.n_layers): model.layers[i].w2 = read_tensor(f'blk.{i}.ffn_down', p.hidden_dim, p.dim)
    for i in range(p.n_layers): model.layers[i].w3 = read_tensor(f'blk.{i}.ffn_up', p.dim, p.hidden_dim)
    model.output_norm = read_tensor('output_norm', p.dim)
    read_tensor('freq_cis_real', p.seq_len, p.head_size // 2) # Discard
    read_tensor('freq_cis_imag', p.seq_len, p.head_size // 2) # Discard
    
    if shared_weights:
        # copy(np.ascontiguousarray(numpy(w.token_embedding_table).transpose()), w.wcls)
        model.output = model.tok_embeddings
        # model.output = new_tensor(model.ctx, copy(np.ascontiguousarray(numpy(w.tok_embeddings)), w.output)
    else:
        model.output = read_tensor('output', p.dim, p.vocab_size)

    return model

def read_format(f, fmt): return struct.unpack_from(fmt, f.read(struct.calcsize(fmt)))

def read_vocab(tokenizer_model: Path, config: Config) -> Vocabulary:
    with tokenizer_model.open('rb') as fd:
        max_token_length = read_format(fd, '<i')[0]
        vocab = []
        vocab_scores = []
        for i in range(config.vocab_size):
            vocab_scores.append(read_format(fd, '<f')[0])
            vocab.append(fd.read(read_format(fd, '<i')[0]).decode('utf-8'))

        sorted_vocab = [TokenIndex(str, i) for i, str in enumerate(vocab)]
        sorted_vocab.sort(key=lambda x: x.str)

        return Vocabulary(config.vocab_size, vocab, vocab_scores, max_token_length, sorted_vocab)

# ----------------------------------------------------------------------------
# sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

def sample(probabilities: TensorLike, n) -> int:
    # probabilities = get_np(probabilities)
    assert probabilities.shape == (n, 1)
    # sample index from probabilities (they must sum to 1!)
    r = np.random.rand()
    cdf = 0.0
    for i in range(n):
        cdf += probabilities[i]
        if r < cdf:
            return i
    return n - 1 # in case of rounding errors

def sample_topp(probabilities: TensorLike, n: int, topp: float, probindex: list[ProbIndex]) -> int:
    # top-p sampling (or "nucleus sampling") samples from the smallest set of
    # tokens that exceed probability topp. This way we never sample tokens that
    # have very low probabilities and are less likely to go "off the rails".

    # quicksort indices in descending order of probabilities
    for i in range(n):
        probindex[i].prob = probabilities[i]
        probindex[i].index = i
    # probindex = [ProbIndex(probabilities[i], i) for i in range(n)]
    probindex.sort(key=lambda x: x.prob, reverse=True)

    # truncate the list where cumulative probability exceeds topp
    cumulative_prob = 0.0
    last_idx = 0
    for i in range(n):
        cumulative_prob += probindex[i].prob
        if cumulative_prob > topp:
            last_idx = i
            break # we've exceeded topp by including last_idx

    # sample from the truncated list
    r = np.random.rand() * cumulative_prob
    cdf = 0.0
    for i in range(last_idx+1):
        cdf += probindex[i].prob
        if r < cdf:
            return probindex[i].index
    return probindex[last_idx].index # in case of rounding errors

def llama_token_bos(): return 1
def llama_token_eos(): return 2
def llama_token_nl(): return 13

def run(
        model: Path,
        tokenizer_model: Path,
        prompt: Optional[str] = None,
        steps: int = 128,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        auto_stop = True,
        debug = True,
        n_threads: int = 6,
        skip_unused: bool = True, # Optimization that skips unused computations during prompt processing (where we only care about the last token's logits really)
        topp: float = 0.9): # top-p in nucleus sampling

    if debug: debug_ggml_asserts()

    np.random.seed(seed)

    fd = os.open(model.as_posix(), os.O_RDONLY)
    mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)

    config = Config(*read_format(mm, '<iiiiiii'))
    shared_weights = config.vocab_size > 0
    config.vocab_size = int(math.fabs(config.vocab_size))

    print(config)

    # tensor_type = lib.GGML_TYPE_Q5_K
    # tensor_type = lib.GGML_TYPE_F32
    # tensor_type = lib.GGML_TYPE_F16

    # weights = LlamaWeights(config, tensor_type, shared_weights)
    model = read_llama2c_model(mm, config, shared_weights)
    # checkpoint_init_weights(mm, config, weights)
    
    mm.close()
    os.close(fd)

    vocab = read_vocab(tokenizer_model, config)
    kv_cache = KVCache(config,
                    #  lib.GGML_TYPE_Q4_0)
                    #  lib.GGML_TYPE_Q5_K)
                    lib.GGML_TYPE_F16)
                    #  tensor_type)
    probindex = [ProbIndex(-1.0, -1) for i in range(config.vocab_size)]

    # if METAL:
    #     state.ctx_metal = lib.ggml_metal_init(1)
    #     lib.ggml_metal_add_buffer(state.ctx_metal, "data", data_ptr, data_size, max_size)

    # process the prompt, if any
    prompt_tokens = None
    if prompt is not None:
        print('Processing prompt...')
        prompt_tokens = bpe_encode(prompt, vocab)
        steps += len(prompt_tokens)

    print('prompt_tokens', prompt_tokens)

    # start the main loop
    start = 0   # used to time our code, only initialized after first iteration
    next = None # will store the next token in the sequence
    pos = 0     # position in the sequence
    
    # init with BOS, as done in Llama-2 sentencepiece tokenizer
    token = llama_token_bos()

    def pick_token(result: GraphResult) -> int:
        if result.argmax is not None:
            argmax = numpy(result.argmax)
            assert argmax.shape[1] == 1, f'Bad argmax shape: {argmax.shape} != {(1, 1)}'
            return argmax[-1, 0]
        else:
            assert result.logits is not None
            logits = numpy(result.logits)
            assert logits.shape[0] == config.vocab_size, f'Bad logits shape: {logits.shape} != ({config.vocab_size}, N)'
            logits = logits[:, -1] # only care about the last token's logits
            if (topp <= 0):
                # simply sample from the predicted probability distribution
                return sample(logits, config.vocab_size)
            else:
                # top-p (nucleus) sampling, clamping the least likely tokens to zero
                return sample_topp(logits, config.vocab_size, topp, probindex)

    def mk_params(tokens):
        return EvalParams(
            n_past=pos,
            n_threads=n_threads,
            temperature=temperature,
            all_logits=not skip_unused,
            tokens=np.array(tokens, dtype=np.int32))
    
    lctx = LlamaContext(config, kv_cache, model)

    if prompt_tokens:
        print("prompt token count: ", len(prompt_tokens), file=sys.stderr)
        prompt_tokens.insert(0, token)

        res = lctx.llama_eval(mk_params(prompt_tokens))
        pos += len(prompt_tokens)
        token = pick_token(res)
        
        sys.stdout.write(''.join([vocab.vocab[t] for t in [*prompt_tokens[1:], token]]))
        sys.stdout.flush()
        # token = llama_token_bos()

    while (pos < steps):
        assert pos < config.seq_len, f'pos {pos} >= seq_len {config.seq_len} (TODO: add support for context compression)'

        result = lctx.llama_eval(mk_params([token]))
        pos += 1
        next = pick_token(result)

        # data-dependent terminating condition: the BOS (1) token delimits sequences
        if auto_stop:
            if next == llama_token_bos(): break
            if next == llama_token_eos():
                print(" [end of text]\n", file=sys.stderr)
                break

        # following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
        if token == 1 and vocab.vocab[next][0] == ' ':
        # if embd[-1] == 1 and vocab.vocab[next][0] == ' ':
            token_str = vocab.vocab[next][1:]
        else:
            token_str = vocab.vocab[next]
        sys.stdout.write(token_str)
        sys.stdout.flush()

        # embd.append(next)
        token = next

        # init the timer here because the first iteration can be slower
        if start == 0: start = time.time()

    print("")

    # report achieved tok/s (pos-1 because the timer starts after first iteration)
    if pos > 1:
        end = time.time()
        print(f"achieved tok/s: {(pos-1) / float(end-start)}", file=sys.stderr)

if __name__ == '__main__':
    CLI(run)
