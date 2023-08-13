# Ported from llama2.c
# https://github.com/ochafik/llama2.c/blob/master/run.c

# cmake ../../../llama.cpp -B /tmp/llama_release -DLLAMA_METAL=1 -DBUILD_SHARED_LIBS=1 -DCMAKE_BUILD_TYPE=Release && ( cd /tmp/llama_release && make -j )
# import os; os.environ['GGML_LIBRARY'] = '/tmp/llama_release/libggml_shared.dylib'

# cmake ../../../llama.cpp -B /tmp/llama_debug -DLLAMA_METAL=1 -DBUILD_SHARED_LIBS=1 -DCMAKE_BUILD_TYPE=Debug && ( cd /tmp/llama_debug && make -j )
import os; os.environ['GGML_LIBRARY'] = '/tmp/llama_debug/libggml_shared.dylib'


from ggml import lib, ffi
from ggml.utils import init, numpy, copy
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict, Any, Callable, TypeVar, Generic
from jsonargparse import CLI
from pathlib import Path
import struct, os, mmap, math, sys
import itertools

Tensor = ffi.CData
Context = ffi.CData

# ----------------------------------------------------------------------------
# Transformer, RunState and Vocabulary structs, and related memory management

__tensor_factories = {
    1: lib.ggml_new_tensor_1d,
    2: lib.ggml_new_tensor_2d,
    3: lib.ggml_new_tensor_3d,
    4: lib.ggml_new_tensor_4d,
}

def _new_tensor(ctx, shape, type):
    factory = __tensor_factories[len(shape)]
    if factory:
        return factory(ctx, type, *shape)
    
    dims = ffi.new('int[]', len(shape))
    for i, dim in enumerate(shape): dims[i] = dim
    return lib.ggml_new_tensor(ctx, type, len(shape), dims)

@dataclass
class Config:
    dim: int # transformer dimension
    hidden_dim: int # for ffn layers
    n_layers: int # number of layers
    n_heads: int # number of query heads
    n_kv_heads: int # number of key/value heads (can be < query heads because of multiquery)
    vocab_size: int # vocabulary size, usually 256 (byte-level)
    seq_len: int # max sequence length

@dataclass
class TransformerWeights:
    # token embedding table
    token_embedding_table: Tensor    # (vocab_size, dim)
    # weights for rmsnorms
    rms_att_weight: list[Tensor] # (layer, dim) rmsnorm weights
    rms_ffn_weight: list[Tensor] # (layer, dim)
    # weights for matmuls
    wq: list[Tensor] # (layer, dim, dim)
    wk: list[Tensor] # (layer, dim, dim)
    wv: list[Tensor] # (layer, dim, dim)
    wo: list[Tensor] # (layer, dim, dim)
    # weights for ffn
    w1: list[Tensor] # (layer, hidden_dim, dim)
    w2: list[Tensor] # (layer, dim, hidden_dim)
    w3: list[Tensor] # (layer, hidden_dim, dim)
    # final rmsnorm
    rms_final_weight: Tensor # (dim,)
    # freq_cis for RoPE relatively positional embeddings
    freq_cis_real: Tensor # (seq_len, head_size/2)
    freq_cis_imag: Tensor # (seq_len, head_size/2)
    # (optional) classifier weights for the logits, on the last layer
    wcls: Tensor

    def __init__(self, p: Config, ctx: Context, type: int, shared_weights: bool):
        self.token_embedding_table = _new_tensor(ctx, (p.vocab_size, p.dim), type)
        self.rms_att_weight = [_new_tensor(ctx, (p.dim,), type) for _ in range(p.n_layers)]
        self.rms_ffn_weight = [_new_tensor(ctx, (p.dim,), type) for _ in range(p.n_layers)]
        self.wq = [_new_tensor(ctx, (p.dim, p.dim), type) for _ in range(p.n_layers)]
        self.wk = [_new_tensor(ctx, (p.dim, p.dim), type) for _ in range(p.n_layers)]
        self.wv = [_new_tensor(ctx, (p.dim, p.dim), type) for _ in range(p.n_layers)]
        self.wo = [_new_tensor(ctx, (p.dim, p.dim), type) for _ in range(p.n_layers)]
        self.w1 = [_new_tensor(ctx, (p.hidden_dim, p.dim), type) for _ in range(p.n_layers)]
        self.w2 = [_new_tensor(ctx, (p.dim, p.hidden_dim), type) for _ in range(p.n_layers)]
        self.w3 = [_new_tensor(ctx, (p.hidden_dim, p.dim), type) for _ in range(p.n_layers)]
        self.rms_final_weight = _new_tensor(ctx, (p.dim,), type)
        self.freq_cis_real = _new_tensor(ctx, (p.seq_len, p.n_heads // 2), type)
        self.freq_cis_imag = _new_tensor(ctx, (p.seq_len, p.n_heads // 2), type)
        self.wcls = _new_tensor(ctx, (p.dim, p.vocab_size), type) if shared_weights else self.token_embedding_table

# struct used when sorting probabilities during top-p sampling
@dataclass
class ProbIndex:
    prob: float
    index: int

@dataclass
class RunState:
    # current wave of activations
    x: Tensor # activation at current time stamp (dim,)
    x_batch_input: Tensor # activations at the previous layer for the current batch (seq_len, dim)
    xb: Tensor # same, but inside a residual branch (dim,)
    xb2: Tensor # an additional buffer just for convenience (dim,)
    hb: Tensor # buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Tensor # buffer for hidden dimension in the ffn (hidden_dim,)
    max_ha1: Tensor # buffer for max activations of hidden dimension in the ffn (layer, hidden_dim)
    max_ha2: Tensor # buffer for max activations of hidden dimension in the ffn (layer, hidden_dim)
    q: Tensor # query (dim,)
    k: Tensor # key (dim,)
    v: Tensor # value (dim,)
    att: Tensor # buffer for scores/attention values (n_heads, seq_len)
    logits: Tensor # output logits
    probindex: list[ProbIndex] # buffer used in top-p sampling
    # kv cache
    key_cache: Tensor   # (layer, seq_len, dim)
    value_cache: Tensor # (layer, seq_len, dim)

    def __init__(self, config: Config, ctx: int, tensor_type: int):
        self.x = _new_tensor(ctx, (config.dim,), tensor_type)
        self.xb = _new_tensor(ctx, (config.dim,), tensor_type)
        self.xb2 = _new_tensor(ctx, (config.dim,), tensor_type)
        self.hb = _new_tensor(ctx, (config.hidden_dim,), tensor_type)
        self.hb2 = _new_tensor(ctx, (config.hidden_dim,), tensor_type)
        self.q = _new_tensor(ctx, (config.dim,), tensor_type)
        self.k = _new_tensor(ctx, (config.dim,), tensor_type)
        self.v = _new_tensor(ctx, (config.dim,), tensor_type)
        self.att = _new_tensor(ctx, (config.n_heads, config.seq_len), tensor_type)
        self.logits = _new_tensor(ctx, (config.vocab_size,), tensor_type)
        # self.probindex = [_new_probindex(ctx) for _ in range(config.seq_len)]
        self.key_cache = _new_tensor(ctx, (config.n_layers, config.seq_len, config.dim), tensor_type)
        self.value_cache = _new_tensor(ctx, (config.n_layers, config.seq_len, config.dim), tensor_type)

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

ScalarTensor = TypeVar('ScalarTensor')
Tensor = TypeVar('Tensor')

rms_norm_eps = 1e-5
rope_freq_base  = 10000.0
rope_freq_scale = 1.0

def build_transformer_graph(ctx, token: ScalarTensor, pos: int, p: Config, s: RunState, w: TransformerWeights):

    # a few convenience variables
    dim = p.dim
    hidden_dim =  p.hidden_dim
    head_size = dim // p.n_heads

    # copy the token embedding into x
    x = lib.ggml_get_rows(ctx, w.token_embedding_table, token)


    # # pluck out the "pos" row of freq_cis_real and freq_cis_imag
    # freq_cis_real_row = w.freq_cis_real + pos * head_size / 2
    # freq_cis_imag_row = w.freq_cis_imag + pos * head_size / 2

    # forward all the layers
    for l in range(p.n_layers):

        # attention rmsnorm
        x = lib.ggml_rms_norm(ctx, x, rms_norm_eps)
        x = lib.ggml_mul(ctx, x, w.rms_att_weight[l])
        
        # qkv matmuls for this position
        tmpk = lib.ggml_mul_mat(ctx, w.wk[l], x)
        tmpq = lib.ggml_mul_mat(ctx, w.wq[l], x)
        Kcur = lib.ggml_rope_custom_inplace(ctx, lib.ggml_reshape_3d(ctx, tmpk, n_embd_head, n_head_kv, N), n_past, n_embd_head, 0, 0, rope_freq_base, rope_freq_scale)
        Qcur = lib.ggml_rope_custom_inplace(ctx, lib.ggml_reshape_3d(ctx, tmpq, n_embd_head, n_head, N),    n_past, n_embd_head, 0, 0, rope_freq_base, rope_freq_scale)


        tmpv = lib.ggml_mul_mat(ctx, w.wv[l], x)
                
        Vcur = lib.ggml_transpose(ctx, lib.ggml_reshape_2d(ctx, tmpv, n_embd_gqa, N))
                
        k = lib.ggml_view_1d(ctx, s.key_cache, N*n_embd_gqa, (lib.ggml_element_size(kv_self.k)*n_embd_gqa)*(il*n_ctx + n_past))

#         v = lib.ggml_view_2d(ctx, s.value_cache, N, n_embd_gqa,
#                         (   n_ctx)*lib.ggml_element_size(s.value_cache),
#                          (l*n_ctx)*lib.ggml_element_size(s.value_cache)*n_embd_gqa + n_past*lib.ggml_element_size(s.value_cache))
                
#         # important: storing RoPE-ed version of K in the KV cache!
#         lib.ggml_build_forward_expand(gf, lib.ggml_cpy(ctx, Kcur, k))
#         lib.ggml_build_forward_expand(gf, lib.ggml_cpy(ctx, Vcur, v))
        

#         Q = lib.ggml_permute(ctx,
#                     Qcur,
#                     0, 2, 1, 3)
        
#         K = lib.ggml_permute(ctx,
#                     lib.ggml_reshape_3d(ctx,
#                         lib.ggml_view_1d(ctx, kv_self.k, (n_past + N)*n_embd_gqa, il*n_ctx*lib.ggml_element_size(kv_self.k)*n_embd_gqa),
#                         n_embd_head, n_head_kv, n_past + N),
#                     0, 2, 1, 3)
        
#         # K * Q
#         KQ = lib.ggml_mul_mat(ctx, K, Q)
        
#         # KQ_scaled = KQ / sqrt(n_embd_head)
#         # KQ_scaled shape [n_past + N, N, n_head, 1]
#         KQ_scaled = lib.ggml_scale_inplace(ctx, KQ, KQ_scale)
        
#         # KQ_masked = mask_past(KQ_scaled)
#         KQ_masked = lib.ggml_diag_mask_inf_inplace(ctx, KQ_scaled, n_past)
        
#         # KQ = soft_max(KQ_masked)
#         KQ_soft_max = lib.ggml_soft_max_inplace(ctx, KQ_masked)

#         # split cached V into n_head heads
#         V = lib.ggml_view_3d(ctx, kv_self.v,
#                     n_past + N, n_embd_head, n_head_kv,
#                     n_ctx*lib.ggml_element_size(kv_self.v),
#                     n_ctx*lib.ggml_element_size(kv_self.v)*n_embd_head,
#                     n_ctx*lib.ggml_element_size(kv_self.v)*n_embd_gqa*il)
        

#         KQV = lib.ggml_mul_mat(ctx, V, KQ_soft_max)
        
#         # KQV_merged = KQV.permute(0, 2, 1, 3)
#         KQV_merged = lib.ggml_permute(ctx, KQV, 0, 2, 1, 3)

#         # x = KQV_merged.contiguous().view(n_embd, N)
#         x = lib.ggml_cpy(ctx,
#                 KQV_merged,
#                 lib.ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, N))

#         # projection (no bias)
#         # final matmul to get the output of the attention
#         s.xb2 = lib.ggml_mul_mat(ctx, s.xb, w.wo[l])

#         # residual connection back into x
#         accum(x, s.xb2, dim)

#         struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);
#         offload_func(inpFF);
#         ggml_set_name(inpFF, "inpFF");

#         // feed-forward network
#         {
#             // norm
#             {
#                 cur = ggml_rms_norm(ctx0, inpFF, rms_norm_eps);
#                 offload_func(cur);
#                 ggml_set_name(cur, "rms_norm_1");

#                 // cur = cur*ffn_norm(broadcasted)
#                 cur = ggml_mul(ctx0, cur, model.layers[il].ffn_norm);
#                 offload_func(cur);
#                 ggml_set_name(cur, "ffn_norm");
#             }

#             struct ggml_tensor * tmp = ggml_mul_mat(ctx0,
#                     model.layers[il].w3,
#                     cur);
#             offload_func(tmp);
#             ggml_set_name(tmp, "result_w3");

#             cur = ggml_mul_mat(ctx0,
#                     model.layers[il].w1,
#                     cur);
#             offload_func(cur);
#             ggml_set_name(cur, "result_w1");

#             // SILU activation
#             cur = ggml_silu(ctx0, cur);
#             offload_func(cur);
#             ggml_set_name(cur, "silu");

#             cur = ggml_mul(ctx0, cur, tmp);
#             offload_func(cur);
#             ggml_set_name(cur, "silu_x_result_w3");

#             cur = ggml_mul_mat(ctx0,
#                     model.layers[il].w2,
#                     cur);
#             offload_func(cur);
#             ggml_set_name(cur, "result_w2");
#         }

#         cur = ggml_add(ctx0, cur, inpFF);
#         offload_func(cur);
#         ggml_set_name(cur, "inpFF_+_result_w2");

#         // input for next layer
#         inpL = cur;
#     }

#     lctx.use_buf(ctx0, 0);

#     // norm
#     {
#         cur = ggml_rms_norm(ctx0, inpL, rms_norm_eps);
#         offload_func_nr(cur);
#         ggml_set_name(cur, "rms_norm_2");

#         // cur = cur*norm(broadcasted)
#         cur = ggml_mul(ctx0, cur, model.norm);
#         // offload_func_nr(cur); // TODO CPU + GPU mirrored backend
#         ggml_set_name(cur, "result_norm");
#     }

#     // lm_head
#     cur = ggml_mul_mat(ctx0, model.output, cur);
#     ggml_set_name(cur, "result_output");

#     lctx.use_buf(ctx0, -1);

#     // logits -> probs
#     //cur = ggml_soft_max_inplace(ctx0, cur);

#     ggml_build_forward_expand(gf, cur);


#         # ffn rmsnorm
#         rmsnorm(s.xb, x, w.rms_ffn_weight + l*dim, dim)

#         # Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
#         # first calculate self.w1(x) and self.w3(x)
#         matmul(s.hb, s.xb, w.w1 + l*dim*hidden_dim, dim, hidden_dim)
#         matmul(s.hb2, s.xb, w.w3 + l*dim*hidden_dim, dim, hidden_dim)

#         # F.silu silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
#         for i in range(hidden_dim):
#             s.hb[i] = s.hb[i] * (1.0f / (1.0f + expf(-s.hb[i])))

#         # elementwise multiply with w3(x)
#         for i in range(hidden_dim):
#             s.hb[i] = s.hb[i] * s.hb2[i]

#         # final matmul to get the output of the ffn
#         matmul(s.xb, s.hb, w.w2 + l*dim*hidden_dim, hidden_dim, dim)

#         # residual connection
#         accum(x, s.xb, dim)

#     # final rmsnorm
#     rmsnorm(x, x, w.rms_final_weight, dim)

#     # classifier into logits
#     matmul(s.logits, x, w.wcls, p.dim, p.vocab_size)

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

def read_tensor(f, tensor):
    shape = tuple([tensor.ne[i] for i in range(tensor.n_dims)])
    nbytes = np.prod(shape) * ffi.sizeof('float')
    copy(np.frombuffer(f.read(nbytes), dtype=np.float32).reshape(shape), tensor)

def run(
        model: Path,
        tokenizer_model: Path,
        prompt: Optional[str] = None,
        steps: int = 256,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        n_threads: int = 1,
        topp: float = 0.9): # top-p in nucleus sampling

    fd = os.open(model.as_posix(), os.O_RDONLY)
    mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
    
    def read_format(f, fmt): return struct.unpack_from(fmt, f.read(struct.calcsize(fmt)))

    config = Config(*read_format(mm, '<iiiiiii'))
    shared_weights = config.vocab_size > 0
    config.vocab_size = int(math.fabs(config.vocab_size))

    print(config)

    ctx = init(mem_size=1024*1024*1024)

    # type = lib.GGML_TYPE_Q5_K
    tensor_type = lib.GGML_TYPE_F32

    weights = TransformerWeights(config, ctx, tensor_type, shared_weights)
    read_tensor(mm, weights.token_embedding_table)
    for i in range(config.n_layers): read_tensor(mm, weights.rms_att_weight[i])
    for i in range(config.n_layers): read_tensor(mm, weights.wq[i])
    for i in range(config.n_layers): read_tensor(mm, weights.wk[i])
    for i in range(config.n_layers): read_tensor(mm, weights.wv[i])
    for i in range(config.n_layers): read_tensor(mm, weights.wo[i])
    for i in range(config.n_layers): read_tensor(mm, weights.rms_ffn_weight[i])
    for i in range(config.n_layers): read_tensor(mm, weights.w1[i])
    for i in range(config.n_layers): read_tensor(mm, weights.w2[i])
    for i in range(config.n_layers): read_tensor(mm, weights.w3[i])
    read_tensor(mm, weights.rms_final_weight)
    read_tensor(mm, weights.freq_cis_real)
    read_tensor(mm, weights.freq_cis_imag)
    if shared_weights:
        weights.wcls = weights.token_embedding_table

    # print('FINISHED READING TransformerWeights')
    mm.close()
    os.close(fd)

    with tokenizer_model.open('rb') as fd:
        max_token_length = read_format(fd, '<i')[0]
        vocab = []
        vocab_scores = []
        for i in range(config.vocab_size):
            vocab_scores.append(read_format(fd, '<f')[0])
            vocab.append(fd.read(read_format(fd, '<i')[0]).decode('utf-8'))

        sorted_vocab = [TokenIndex(str, i) for i, str in enumerate(vocab)]
        sorted_vocab.sort(key=lambda x: x.str)

        v = Vocabulary(config.vocab_size, vocab, vocab_scores, max_token_length, sorted_vocab)
        # print('FINISHED READING Vocabulary')

    state = RunState(config, ctx, tensor_type)
    # print('FINISHED BUILDING RunState')

    # process the prompt, if any
    prompt_tokens = None
    if prompt is not None:
        prompt_tokens = bpe_encode(prompt, v)
        steps += len(prompt_tokens)

    print('FINISHED PROCESSING PROMPT')
    print(prompt_tokens)

    # start the main loop
    start = 0  # used to time our code, only initialized after first iteration
    next = None        # will store the next token in the sequence
    token = lib.ggml_new_i32(ctx, 1)   # init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
    pos = 0     # position in the sequence

    gf = build_transformer_graph(ctx, token, pos, config, state, weights)
    lib.ggml_build_forward_expand(gf, state.logits)

    while (pos < steps):
        # forward the transformer to get logits for the next token
        # TODO: actually while we're processing the prompt tokens, we can just
        # build forward expand on the last layer's kv cache write and save some
        # operations. But that would mean creating
        # a different graph so might be counteproductive.
        lib.ggml_graph_compute_with_ctx(ctx, gf, n_threads)

        # advance the state state machine
        if(pos < len(prompt_tokens)):
            # if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos]
        else:
            # sample the next token
            if (temperature == 0.0):
                # greedy argmax sampling: take the token with the highest probability
                next = argmax(state.logits, config.vocab_size)
            else:
                # apply the temperature to the logits
                for q in range(config.vocab_size): state.logits[q] /= temperature
                # apply softmax to the logits to get the probabilities for next token
                softmax(state.logits, config.vocab_size)
                # we sample from this distribution to get the next token
                if (topp <= 0):
                    # simply sample from the predicted probability distribution
                    next = sample(state.logits, config.vocab_size)
                else:
                    # top-p (nucleus) sampling, clamping the least likely tokens to zero
                    next = sample_topp(state.logits, config.vocab_size, topp, state.probindex)

        pos += 1

        # data-dependent terminating condition: the BOS (1) token delimits sequences
        if (next == 1): break

        # following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
        token_str = v.vocab[next]+1 if (token == 1 and v.vocab[next][0] == ' ') else v.vocab[next]
        sys.stdout.write(token_str)
        sys.stdout.flush()
        lib.ggml_set_i32(token, next)

        # init the timer here because the first iteration can be slower
        # if (start == 0): start = time_in_ms(); }

    print("")

if __name__ == '__main__':
    CLI(run)
    