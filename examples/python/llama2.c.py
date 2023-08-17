# Ported from llama2.c
# https://github.com/karpathy/llama2.c/blob/master/run.c
# Use faster tokenization, and naively allocate buffers in GGML / wrap them to numpy

# rm -fR /tmp/llama_release ; cmake ../../../llama.cpp -B /tmp/llama_release -DCMAKE_C_FLAGS=-Ofast -DLLAMA_NATIVE=1 -DLLAMA_LTO=1 -DLLAMA_METAL=1 -DBUILD_SHARED_LIBS=1 -DLLAMA_BUILD_EXAMPLES=0 -DLLAMA_BUILD_TESTS=0 -DCMAKE_BUILD_TYPE=Release && ( cd /tmp/llama_release && make -j )
import os; os.environ['GGML_LIBRARY'] = '/tmp/llama_release/libggml_shared.dylib'

# cmake ../../../llama.cpp -B /tmp/llama_debug -DLLAMA_METAL=1 -DBUILD_SHARED_LIBS=1 -DLLAMA_BUILD_EXAMPLES=0 -DLLAMA_BUILD_TESTS=0 -DCMAKE_BUILD_TYPE=Debug && ( cd /tmp/llama_debug && make -j )
# import os; os.environ['GGML_LIBRARY'] = '/tmp/llama_debug/libggml_shared.dylib'

# python llama2.c.py ~/AI/Models/llama2.c.stories15M.bin ../../../llama2.c/tokenizer.bin --prompt "Hello, world"

from ggml import lib, ffi
from ggml.utils import init, numpy, copy, debug_ggml_asserts
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict, Any, Callable, TypeVar, Generic
from jsonargparse import CLI
from pathlib import Path
import struct, os, mmap, math, sys, time, inspect, signal, ctypes, re
from collections import namedtuple
import scipy as sp

GGML_USE_MPI=False
LLAMA_USE_ALLOCATOR=False
GGML_USE_METAL = os.environ.get('GGML_USE_METAL', '0') == '1'
LLAMA_USE_SCRATCH = os.environ.get('LLAMA_USE_SCRATCH', '0') == '1'

Tensor = ffi.CData
Context = ffi.CData
TensorLike = Union[Tensor, np.ndarray]

def _get_shape(x: TensorLike): return x.shape if isinstance(x, np.ndarray) else tuple([x.ne[i] for i in range(x.n_dims)])

# ----------------------------------------------------------------------------
# Transformer, RunState and Vocabulary structs, and related memory management

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

class TransformerWeights:
    def __init__(self, p: Config, ctx: Context, type: int, shared_weights: bool):
        # token embedding table
        self.token_embedding_table = lib.ggml_new_tensor_2d(ctx, type, p.dim, p.vocab_size)
        # weights for rmsnorms
        self.rms_att_weight = [lib.ggml_new_tensor_1d(ctx, type, p.dim) for _ in range(p.n_layers)]
        self.rms_ffn_weight = [lib.ggml_new_tensor_1d(ctx, type, p.dim) for _ in range(p.n_layers)]
        # weights for matmuls
        self.wq = [lib.ggml_new_tensor_2d(ctx, type, p.dim, p.dim) for _ in range(p.n_layers)]
        self.wk = [lib.ggml_new_tensor_2d(ctx, type, p.dim, p.dim) for _ in range(p.n_layers)]
        self.wv = [lib.ggml_new_tensor_2d(ctx, type, p.dim, p.dim) for _ in range(p.n_layers)]
        self.wo = [lib.ggml_new_tensor_2d(ctx, type, p.dim, p.dim) for _ in range(p.n_layers)]
        # weights for ffn
        self.w1 = [lib.ggml_new_tensor_2d(ctx, type, p.dim, p.hidden_dim) for _ in range(p.n_layers)]
        self.w2 = [lib.ggml_new_tensor_2d(ctx, type, p.hidden_dim, p.dim) for _ in range(p.n_layers)]
        self.w3 = [lib.ggml_new_tensor_2d(ctx, type, p.dim, p.hidden_dim) for _ in range(p.n_layers)]
        # final rmsnorm
        self.rms_final_weight = lib.ggml_new_tensor_1d(ctx, type, p.dim)
        # freq_cis for RoPE relatively positional embeddings
        self.freq_cis_real = lib.ggml_new_tensor_2d(ctx, type, p.seq_len, p.head_size // 2)
        self.freq_cis_imag = lib.ggml_new_tensor_2d(ctx, type, p.seq_len, p.head_size // 2)
        # (optional) classifier weights for the logits, on the last layer
        self.wcls = lib.ggml_new_tensor_2d(ctx, type, p.dim, p.vocab_size)
        self.shared_weights = shared_weights


# struct used when sorting probabilities during top-p sampling
@dataclass
class ProbIndex:
    prob: float
    index: int

# current wave of activations
@dataclass
class RunState:
    def __init__(self, config: Config, ctx: int, tensor_type: int):
        # kv cache
        self.key_cache = lib.ggml_new_tensor_3d(ctx, tensor_type, config.n_layers, config.seq_len, config.dim)
        self.value_cache = lib.ggml_new_tensor_3d(ctx, tensor_type, config.n_layers, config.seq_len, config.dim)
        self.cache_length = 0

        self.probindex = [ProbIndex(-1.0, -1) for i in range(config.vocab_size)]

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

def llama_build_graph(
        p: Config,
        s: RunState,
        w: TransformerWeights,
        tokens: Optional[list[int]],
        embd: Optional[np.ndarray],
        n_tokens: int,
        n_past: int,
        # TODO: options:
        # - No output (just kv-cache building) if Temperature = None
        # - Softmax(Logits / Temperature) if Temperature != 0
        # - Argmax(Logits) if Temperature == 0
        # - ?? Raw embeddings
        # - ? Raw logits (useful for distillation?)
        # - Other sampling methods? Mirostat, etc
        compute_logits: bool = True,
        temperature: Optional[float] = None):
    assert ((not tokens and embd) or (tokens and not embd))

    N = n_tokens
    n_gqa = p.n_heads // p.n_kv_heads
    n_embd = p.dim
    n_embd_gqa = p.dim // n_gqa
    n_embd_head = p.head_size
    n_layer = p.n_layers
    n_ctx = p.seq_len
    n_head = p.n_heads
    n_head_kv = p.n_kv_heads
    
    params = ffi.new('struct ggml_init_params*')
    params.mem_size = 100*1024*1024
    params.mem_buffer = ffi.NULL
    params.no_alloc = False
    ctx0 = lib.ggml_init(params[0])
    try:

        gf = lib.ggml_new_graph(ctx0)

        if tokens:
            inp_tokens = lib.ggml_new_tensor_1d(ctx0, lib.GGML_TYPE_I32, N)
            # copy(np.array(tokens, dtype=np.int32), inp_tokens)
            ffi.memmove(inp_tokens.data, np.array(tokens, dtype=int), N*lib.ggml_element_size(inp_tokens))
            inpL = lib.ggml_get_rows(ctx0, w.token_embedding_table, inp_tokens)
        else:
            inpL = lib.ggml_new_tensor_2d(ctx0, lib.GGML_TYPE_F32, n_embd, N)
            ffi.memmove(inpL.data, ffi.frombuffer(embd), N * n_embd * lib.ggml_element_size(inpL))

        assert _get_shape(inpL) == (n_embd, N), f'Bad input shape: {_get_shape(inpL)} != {(n_embd, N)}'

        KQ_scale = lib.ggml_new_tensor_1d(ctx0, lib.GGML_TYPE_F32, 1)
        lib.ggml_set_f32(KQ_scale, 1.0/math.sqrt(float(n_embd)/n_head))

        for il in range(n_layer):
            inpSA = inpL
            # use_buf(ctx0, 0)
            #  norm
            cur = lib.ggml_rms_norm(ctx0, inpL, rms_norm_eps)
            #  cur = cur*attention_norm(broadcasted)
            cur = lib.ggml_mul(ctx0, cur, w.rms_att_weight[il])

            #  self-attention
            
            #  compute Q and K and RoPE them
            tmpk = lib.ggml_mul_mat(ctx0, w.wk[il], cur)
            tmpq = lib.ggml_mul_mat(ctx0, w.wq[il], cur)
            Kcur = lib.ggml_rope_custom_inplace(ctx0, lib.ggml_reshape_3d(ctx0, tmpk, n_embd_head, n_head_kv, N), n_past, n_embd_head, 0, 0, rope_freq_base, rope_freq_scale)
            Qcur = lib.ggml_rope_custom_inplace(ctx0, lib.ggml_reshape_3d(ctx0, tmpq, n_embd_head, n_head, N),    n_past, n_embd_head, 0, 0, rope_freq_base, rope_freq_scale)

            #  store key and value to memory
            
            #  compute the transposed [N, n_embd] V matrix
            tmpv = lib.ggml_mul_mat(ctx0, w.wv[il], cur)
            Vcur = lib.ggml_transpose(ctx0, lib.ggml_reshape_2d(ctx0, tmpv, n_embd_gqa, N))
            k = lib.ggml_view_1d(ctx0, s.key_cache, N*n_embd_gqa, (lib.ggml_element_size(s.key_cache)*n_embd_gqa)*(il*n_ctx + n_past))
            v = lib.ggml_view_2d(ctx0, s.value_cache, N, n_embd_gqa,
                    (   n_ctx)*lib.ggml_element_size(s.value_cache),
                    (il*n_ctx)*lib.ggml_element_size(s.value_cache)*n_embd_gqa + n_past*lib.ggml_element_size(s.value_cache))
            #  important: storing RoPE-ed version of K in the KV cache!
            lib.ggml_build_forward_expand(gf, lib.ggml_cpy(ctx0, Kcur, k))
            lib.ggml_build_forward_expand(gf, lib.ggml_cpy(ctx0, Vcur, v))

            if not compute_logits and il == n_layer - 1:
                return (gf, None, None)

            Q = lib.ggml_permute(ctx0, Qcur, 0, 2, 1, 3)
            K = lib.ggml_permute(ctx0, lib.ggml_reshape_3d(ctx0,
                            lib.ggml_view_1d(ctx0, s.key_cache, (n_past + N)*n_embd_gqa, il*n_ctx*lib.ggml_element_size(s.key_cache)*n_embd_gqa),
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
            V = lib.ggml_view_3d(ctx0, s.value_cache,
                        n_past + N, n_embd_head, n_head_kv,
                        n_ctx*lib.ggml_element_size(s.value_cache),
                        n_ctx*lib.ggml_element_size(s.value_cache)*n_embd_head,
                        n_ctx*lib.ggml_element_size(s.value_cache)*n_embd_gqa*il)

            if True:
                KQV = lib.ggml_mul_mat(ctx0, V, KQ_soft_max)
            else:
                #  make V contiguous in memory to speed up the matmul, however we waste time on the copy
                #  on M1 this is faster for the perplexity computation, but ~5% slower for the single-token generation
                #  is there a better way?
                V_cont = lib.ggml_cpy(ctx0, V, lib.ggml_new_tensor_3d(ctx0, s.value_cache.type, n_past + N, n_embd_head, n_head))
                KQV = lib.ggml_mul_mat(ctx0, V_cont, KQ_soft_max)
            #  KQV_merged = KQV.permute(0, 2, 1, 3)
            KQV_merged = lib.ggml_permute(ctx0, KQV, 0, 2, 1, 3)
            #  cur = KQV_merged.contiguous().view(n_embd, N)
            cur = lib.ggml_cpy(ctx0,
                    KQV_merged,
                    lib.ggml_new_tensor_2d(ctx0, lib.GGML_TYPE_F32, n_embd, N))
            #  projection (no bias)
            cur = lib.ggml_mul_mat(ctx0, w.wo[il], cur)
            # use_buf(ctx0, 1)
            inpFF = lib.ggml_add(ctx0, cur, inpSA)

            #  feed-forward network
            
            #  norm
            cur = lib.ggml_rms_norm(ctx0, inpFF, rms_norm_eps)
            #  cur = cur*ffn_norm(broadcasted)
            cur = lib.ggml_mul(ctx0, cur, w.rms_ffn_weight[il])
            tmp = lib.ggml_mul_mat(ctx0, w.w3[il], cur)
            cur = lib.ggml_mul_mat(ctx0, w.w1[il], cur)
            #  SILU activation
            cur = lib.ggml_silu(ctx0, cur)
            cur = lib.ggml_mul(ctx0, cur, tmp)
            cur = lib.ggml_mul_mat(ctx0, w.w2[il], cur)
            cur = lib.ggml_add(ctx0, cur, inpFF)
            #  input for next layer
            inpL = cur

        # use_buf(ctx0, 0)

        #  norm
        cur = lib.ggml_rms_norm(ctx0, inpL, rms_norm_eps)

        #  cur = cur*norm(broadcasted)
        cur = lib.ggml_set_name(lib.ggml_mul(ctx0, cur, w.rms_final_weight), ffi.new("char[]", b"result_norm\0"))

        #  lm_head
        cur = lib.ggml_set_name(lib.ggml_mul_mat(ctx0, w.wcls, cur), ffi.new("char[]", b"result_output\0"))
        # use_buf(ctx0, -1)

        #  logits -> probs
        # cur = lib.ggml_soft_max_inplace(ctx0, cur)

        end_offset = 0
        
        if temperature is not None:
            if temperature == 0.0:
                # print('argmax')
                cur = lib.ggml_argmax(ctx0, cur)
                assert _get_shape(cur) == (1, N)
                end_offset += 1
            else:
                cur = lib.ggml_scale_inplace(ctx0, cur, lib.ggml_new_f32(ctx0, 1.0/temperature))
                cur = lib.ggml_soft_max_inplace(ctx0, cur)
                end_offset += 2

        lib.ggml_build_forward_expand(gf, cur)

        res = gf.nodes[gf.n_nodes - 1]
        # assert _get_shape(res) == (p.vocab_size, N)
        # assert ffi.string(res.name).decode('utf-8') == "result_output", ffi.string(res.name)
        
        embeddings = gf.nodes[gf.n_nodes - 2 - end_offset]
        # assert _get_shape(embeddings) == (p.dim, N)
        # assert ffi.string(embeddings.name).decode('utf-8') == "result_norm", ffi.string(embeddings.name)
        
        return (gf, res, embeddings)

    finally:
        lib.ggml_free(ctx0)


def lookup_token(str: str, v: Vocabulary):
    left, right = 0, v.vocab_size - 1
    while (left <= right):
        mid = (left + right) // 2
        midtok = v.sorted_vocab[mid]
        if str == midtok.str: return midtok.id
        elif str < midtok.str: right = mid - 1
        else: left = mid + 1
    return -1

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

def read_tensor(name: str, f, tensor: ffi.CData):
    shape = tuple([tensor.ne[i] for i in range(tensor.n_dims)])
    nbytes = np.prod(shape) * ffi.sizeof('float')
    array = np.frombuffer(f.read(nbytes), dtype=np.float32).reshape(shape)
    copy(array, tensor)

def checkpoint_init_weights(mm, p: Config, w: TransformerWeights):
    read_tensor("token_embedding_table", mm, w.token_embedding_table)
    for i in range(p.n_layers): read_tensor(f'{w.rms_att_weight[i]}', mm, w.rms_att_weight[i])
    for i in range(p.n_layers): read_tensor(f'{w.wq[i]}', mm, w.wq[i])
    for i in range(p.n_layers): read_tensor(f'{w.wk[i]}', mm, w.wk[i])
    for i in range(p.n_layers): read_tensor(f'{w.wv[i]}', mm, w.wv[i])
    for i in range(p.n_layers): read_tensor(f'{w.wo[i]}', mm, w.wo[i])
    for i in range(p.n_layers): read_tensor(f'{w.rms_ffn_weight[i]}', mm, w.rms_ffn_weight[i])
    for i in range(p.n_layers): read_tensor(f'{w.w1[i]}', mm, w.w1[i])
    for i in range(p.n_layers): read_tensor(f'{w.w2[i]}', mm, w.w2[i])
    for i in range(p.n_layers): read_tensor(f'{w.w3[i]}', mm, w.w3[i])
    read_tensor('rms_final_weight', mm, w.rms_final_weight)
    read_tensor('freq_cis_real', mm, w.freq_cis_real)
    read_tensor('freq_cis_imag', mm, w.freq_cis_imag)
    if w.shared_weights:
        copy(np.ascontiguousarray(numpy(w.token_embedding_table).transpose()), w.wcls)
    else:
        read_tensor('wcls', mm, w.wcls)

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

# evaluate the transformer
#
#   - lctx:      llama context
#   - tokens:    new batch of tokens to process
#   - embd       embeddings input
#   - n_tokens   number of tokens
#   - n_past:    the context size so far
#   - n_threads: number of threads to use
#
def llama_eval(
        p: Config,
        s: RunState,
        w: TransformerWeights,
        tokens: Optional[list[int]],
        embd: Optional[np.ndarray],
        n_tokens: int,
        n_past: int,
        n_threads: int,
        compute_logits: bool = True,
        temperature: Optional[float] = None):
    # if GGML_USE_MPI:
    #     ggml_mpi_eval_init(s.ctx_mpi, &n_tokens, &n_past, &n_threads)

    N = n_tokens
    # if LLAMA_USE_ALLOCATOR:
    #     ggml_allocr_reset(lctx.alloc)

    (gf, res, embeddings) = llama_build_graph(p, s, w, tokens, embd, n_tokens, n_past, compute_logits, temperature)

    # if LLAMA_USE_ALLOCATOR:
    #     ggml_allocr_alloc_graph(lctx.alloc, gf)

    # LLAMA_LOG_INFO("graph build time: %.3f ms (%d nodes, %d leafs)\n", (ggml_time_us() - t_start_us)/1000.0, gf->n_nodes, gf->n_leafs)

    # for big prompts, if BLAS is enabled, it is better to use only one thread
    # otherwise, the threads are spin-lock waiting for the BLAS calls and are degrading the performance
    n_threads = 1 if N >= 32 and lib.ggml_cpu_has_blas() and not lib.ggml_cpu_has_gpublas() else n_threads

    
    # if GGML_USE_MPI:
    #     lib.ggml_mpi_graph_compute_pre(lctx.ctx_mpi, gf, n_layer)

    if GGML_USE_METAL:
        if s.ctx_metal and N == 1:
            # TODO: disabled until #2413 is resolved
            #if (!lib.ggml_metal_if_optimized(lctx.ctx_metal)) {
            #    lib.ggml_metal_graph_find_concurrency(lctx.ctx_metal, gf)
            #}
            lib.ggml_metal_set_n_cb     (s.ctx_metal, n_threads)
            lib.ggml_metal_graph_compute(s.ctx_metal, gf)
            lib.ggml_metal_get_tensor   (s.ctx_metal, res)
            # if not lctx.embedding.empty():
            lib.ggml_metal_get_tensor(s.ctx_metal, embeddings)
        else:
            assert False
            # IMPORTANT:
            # Since we don't have efficient Matrix x Matrix Metal multiplication yet, we fallback to vanilla
            # ggml_graph_compute(). It uses Apple's Accelerate CBLAS API which takes advantage of the ANE or the AMX
            # coprocessor.
            #
            # When we implement Matrix x Matrix Metal multiplication, we can avoid this branch.
            # But for now, we have focused only on Matrix x Vector Metal multiplication.
            #
            # TODO: avoid these syncs via shared memory (ref #1696)
            #
            if s.ctx_metal:
                # We need to sync the GPU KV cache with the CPU KV cache
                lib.ggml_metal_get_tensor(s.ctx_metal, kv_self.k)
                lib.ggml_metal_get_tensor(s.ctx_metal, kv_self.v)

            lib.ggml_graph_compute_helper(s.work_buffer, gf, n_threads)
    else:
        # lib.ggml_graph_compute_helper(lctx.work_buffer, gf, n_threads)
        plan = lib.ggml_graph_plan(gf, n_threads)

        # if plan.work_size > 0:
        #     buf.resize(plan.work_size)
        #     plan.work_data = buf.data()

        lib.ggml_graph_compute(gf, ffi.addressof(plan))

    # if GGML_USE_MPI:
    #     lib.ggml_mpi_graph_compute_post(lctx.ctx_mpi, gf, n_layer)

    # update kv token count
    s.cache_length = n_past + N

    # extract logits
    # if logits_all:
    #     logits_out.resize(n_vocab * N)
    #     memcpy(logits_out.data(), (float *) ggml_get_data(res), sizeof(float)*n_vocab*N)
    # else:
    #     # return result for just the last token
    #     logits_out.resize(n_vocab)
    #     memcpy(logits_out.data(), (float *) ggml_get_data(res) + (n_vocab*(N-1)), sizeof(float)*n_vocab)
    # info("res", res)
    logits = numpy(res)[-p.vocab_size*(N-1):] if res is not None else None

    # extract embeddings
    # info("embeddings", embeddings)
    embeddings = numpy(embeddings)[-p.dim*(N-1):] if embeddings is not None else None

    return (logits, embeddings)

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
        topp: float = 0.9): # top-p in nucleus sampling

    if debug: debug_ggml_asserts()

    np.random.seed(seed)

    fd = os.open(model.as_posix(), os.O_RDONLY)
    mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)

    config = Config(*read_format(mm, '<iiiiiii'))
    shared_weights = config.vocab_size > 0
    config.vocab_size = int(math.fabs(config.vocab_size))

    print(config)

    ctx = init(mem_size=100*1024*1024*1024)

    # tensor_type = lib.GGML_TYPE_Q5_K
    tensor_type = lib.GGML_TYPE_F32

    weights = TransformerWeights(config, ctx, tensor_type, shared_weights)
    checkpoint_init_weights(mm, config, weights)
    
    mm.close()
    os.close(fd)

    vocab = read_vocab(tokenizer_model, config)
    state = RunState(config, ctx, tensor_type)

    # if GGML_USE_METAL:
    #     state.ctx_metal = lib.ggml_metal_init(1)
    #     lib.ggml_metal_add_buffer(state.ctx_metal, "data", data_ptr, data_size, max_size)

    # process the prompt, if any
    prompt_tokens = None
    if prompt is not None:
        prompt_tokens = bpe_encode(prompt, vocab)
        steps += len(prompt_tokens)

    print('prompt_tokens', prompt_tokens)

    # start the main loop
    start = 0   # used to time our code, only initialized after first iteration
    next = None # will store the next token in the sequence
    pos = 0     # position in the sequence
    
    # init with BOS, as done in Llama-2 sentencepiece tokenizer
    token = llama_token_bos()

    if prompt_tokens:
        print("prompt token count: ", len(prompt_tokens), file=sys.stderr)
    #     prompt_tokens.insert(0, token)
    #     llama_eval(config, state, weights, prompt_tokens, None, n_tokens=len(prompt_tokens), n_past=pos, n_threads=n_threads)
    #     pos += len(prompt_tokens) - 1
    #     sys.stdout.write(''.join([vocab.vocab[t] for t in prompt_tokens[1:]]))
    #     sys.stdout.flush()
    #     token = prompt_tokens[-1]
    #     # token = llama_token_bos()

    SKIP = os.environ.get('SKIP', '0') == '1'

    while (pos < steps):
        assert pos < config.seq_len, f'pos {pos} >= seq_len {config.seq_len} (TODO: add support for context compression)'

        if SKIP:
            compute_logits = not prompt_tokens or pos >= len(prompt_tokens)# - 1
        else:
            compute_logits = True

        (logits, embeddings) = llama_eval(config, state, weights, [token], None, n_tokens=1, n_past=pos, n_threads=n_threads,
                                          compute_logits=compute_logits,
                                          temperature=temperature)
        
        # advance the state state machine
        if(pos < (len(prompt_tokens) if prompt_tokens else 0)):
            # if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos]
        else:
            # sample the next token
            if (temperature == 0.0):
                # We did greedy argmax sampling: take the token with the highest probability
                assert logits.shape == (1,1), f'Bad shape for argmax result: {logits.shape} = {logits.transpose()}'
                next = logits[0, 0]
                # next = np.argmax(logits)
            else:
                # We applied the temperature & softmax to the logits.
                # we sample from this distribution to get the next token
                if (topp <= 0):
                    # simply sample from the predicted probability distribution
                    next = sample(logits, config.vocab_size)
                else:
                    # top-p (nucleus) sampling, clamping the least likely tokens to zero
                    next = sample_topp(logits, config.vocab_size, topp, state.probindex)

        pos += 1

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
    