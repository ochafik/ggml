# Ported from llama2.c
# https://github.com/ochafik/llama2.c/blob/master/run.c

# cmake ../../../llama.cpp -B /tmp/llama_release -DLLAMA_METAL=1 -DBUILD_SHARED_LIBS=1 -DCMAKE_BUILD_TYPE=Release && ( cd /tmp/llama_release && make -j )
# import os; os.environ['GGML_LIBRARY'] = '/tmp/llama_release/libggml_shared.dylib'

# cmake ../../../llama.cpp -B /tmp/llama_debug -DLLAMA_METAL=1 -DBUILD_SHARED_LIBS=1 -DCMAKE_BUILD_TYPE=Debug && ( cd /tmp/llama_debug && make -j )
import os; os.environ['GGML_LIBRARY'] = '/tmp/llama_debug/libggml_shared.dylib'

# python llama2.c.py ~/AI/Models/llama2.c.stories15M.bin ../../../llama2.c/tokenizer.bin --prompt "Hello, world"

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
    shared_weights: bool # whether we share weights between token embedding and classifier

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
        self.shared_weights = shared_weights
        self.token_embedding_table = _new_tensor(ctx, (p.dim, p.vocab_size), type)
        self.rms_att_weight = [_new_tensor(ctx, (p.dim,), type) for _ in range(p.n_layers)]
        self.rms_ffn_weight = [_new_tensor(ctx, (p.dim,), type) for _ in range(p.n_layers)]
        self.wq = [_new_tensor(ctx, (p.dim, p.dim), type) for _ in range(p.n_layers)]
        self.wk = [_new_tensor(ctx, (p.dim, p.dim), type) for _ in range(p.n_layers)]
        self.wv = [_new_tensor(ctx, (p.dim, p.dim), type) for _ in range(p.n_layers)]
        self.wo = [_new_tensor(ctx, (p.dim, p.dim), type) for _ in range(p.n_layers)]
        self.w1 = [_new_tensor(ctx, (p.dim, p.hidden_dim), type) for _ in range(p.n_layers)]
        self.w2 = [_new_tensor(ctx, (p.hidden_dim, p.dim), type) for _ in range(p.n_layers)]
        self.w3 = [_new_tensor(ctx, (p.dim, p.hidden_dim), type) for _ in range(p.n_layers)]
        self.rms_final_weight = _new_tensor(ctx, (p.dim,), type)
        self.freq_cis_real = _new_tensor(ctx, (p.n_heads // 2, p.seq_len), type)
        self.freq_cis_imag = _new_tensor(ctx, (p.n_heads // 2, p.seq_len), type)
        self.wcls = _new_tensor(ctx, (p.vocab_size, p.dim), type) if shared_weights else self.token_embedding_table

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

def info(x: Tensor, name: str): 
    a = numpy(x)
    print(name, a.shape)#, a)

def build_transformer_graph(ctx, token: ScalarTensor, pos: int, p: Config, s: RunState, w: TransformerWeights):

    # a few convenience variables
    dim = p.dim
    hidden_dim =  p.hidden_dim
    head_size = dim // p.n_heads

    # copy the token embedding into x
    info(w.token_embedding_table, "token_embedding_table")
    info(token, "token")
    # x = lib.ggml_get_rows(ctx, w.token_embedding_table, token)
    s.x = lib.ggml_get_rows(ctx, w.token_embedding_table, token)
    info(s.x, "x")
    assert(numpy(s.x).shape == (dim, 1))


    # # pluck out the "pos" row of freq_cis_real and freq_cis_imag
    # freq_cis_real_row = w.freq_cis_real + pos * head_size / 2
    # freq_cis_imag_row = w.freq_cis_imag + pos * head_size / 2

    # forward all the layers

    n_gqa = p.n_heads // p.n_kv_heads
    assert p.n_heads % p.n_kv_heads == 0
    n_embd_gqa = p.dim // n_gqa
    assert p.dim % n_gqa == 0

    KQ_scale = lib.ggml_new_f32(ctx, 1.0/math.sqrt(float(p.dim)/p.n_heads))
    

    gf = lib.ggml_new_graph(ctx)

    for l in range(p.n_layers):
        # in llama.cpp:
        #   inpSA = inpL,
        #   cur gets rmsnorm, then attention norm
        #   then cur gets projected onto K, Q
        #   ...

        # llama.cpp  vs llama2.c
        #          inpL <-> x
        #        n_past <-> pos
        #  N = n_tokens <-> p.seq_len??
        #    n_embd_gqa <-> ??
        #        n_embd <-> p.dim ?? (in other places like kv_cache_init it's n_embd_gqa. CAREFUL)
        #         n_ctx <-> 1 (one context only)
        #        n_head <-> p.n_heads

        # in llama2.c:
        #   xb gets rmsnorm(x), and it gets projected onto q, k, v
        #   

        # attention rmsnorm
        s.x = lib.ggml_rms_norm(ctx, s.x, rms_norm_eps)
        info(s.x, "s.x")
        info(w.rms_att_weight[l], "rms_att_weight")
        s.x = lib.ggml_mul(ctx, s.x, w.rms_att_weight[l])
        
        # qkv matmuls for this position
        tmpk = lib.ggml_mul_mat(ctx, w.wk[l], s.x)
        tmpq = lib.ggml_mul_mat(ctx, w.wq[l], s.x)
        Kcur = lib.ggml_rope_custom_inplace(ctx, lib.ggml_reshape_3d(ctx, tmpk, p.n_heads, p.n_kv_heads, p.seq_len), pos, p.n_heads, 0, 0, rope_freq_base, rope_freq_scale)
        Qcur = lib.ggml_rope_custom_inplace(ctx, lib.ggml_reshape_3d(ctx, tmpq, p.n_heads, p.n_heads, p.seq_len),    pos, p.n_heads, 0, 0, rope_freq_base, rope_freq_scale)


        tmpv = lib.ggml_mul_mat(ctx, w.wv[l], s.x)
                
        Vcur = lib.ggml_transpose(ctx, lib.ggml_reshape_2d(ctx, tmpv, n_embd_gqa, p.seq_len))
                
        k = lib.ggml_view_1d(ctx, s.key_cache, p.seq_len*n_embd_gqa, (lib.ggml_element_size(s.key_cache)*n_embd_gqa)*(l + pos))

        v = lib.ggml_view_2d(ctx, s.value_cache, p.seq_len, n_embd_gqa,
                                lib.ggml_element_size(s.value_cache),
                              l*lib.ggml_element_size(s.value_cache)*n_embd_gqa + pos*lib.ggml_element_size(s.value_cache))
                
        # important: storing RoPE-ed version of K in the KV cache!
        lib.ggml_build_forward_expand(gf, lib.ggml_cpy(ctx, Kcur, k))
        lib.ggml_build_forward_expand(gf, lib.ggml_cpy(ctx, Vcur, v))
        
        Q = lib.ggml_permute(ctx, Qcur, 0, 2, 1, 3)
        
        K = lib.ggml_permute(ctx,
                    lib.ggml_reshape_3d(ctx,
                        lib.ggml_view_1d(ctx, s.key_cache, (pos + p.seq_len)*n_embd_gqa, l*lib.ggml_element_size(s.key_cache)*n_embd_gqa),
                        p.n_heads, p.n_kv_heads, pos + p.seq_len),
                    0, 2, 1, 3)
        
        # K * Q
        KQ = lib.ggml_mul_mat(ctx, K, Q)
        
        # KQ_scaled = KQ / sqrt(p.n_heads)
        # KQ_scaled shape [pos + N, N, n_head, 1]
        KQ_scaled = lib.ggml_scale_inplace(ctx, KQ, KQ_scale)
        
        # KQ_masked = mask_past(KQ_scaled)
        KQ_masked = lib.ggml_diag_mask_inf_inplace(ctx, KQ_scaled, pos)
        
        # KQ = soft_max(KQ_masked)
        KQ_soft_max = lib.ggml_soft_max_inplace(ctx, KQ_masked)

        # split cached V into n_head heads
        V = lib.ggml_view_3d(ctx, s.value_cache,
                    pos + p.seq_len, p.n_heads, p.n_kv_heads,
                    lib.ggml_element_size(s.value_cache),
                    lib.ggml_element_size(s.value_cache)*p.n_heads,
                    lib.ggml_element_size(s.value_cache)*n_embd_gqa*l)
        

        KQV = lib.ggml_mul_mat(ctx, V, KQ_soft_max)
        
        # KQV_merged = KQV.permute(0, 2, 1, 3)
        KQV_merged = lib.ggml_permute(ctx, KQV, 0, 2, 1, 3)

        # x = KQV_merged.contiguous().view(n_embd, N)
        s.xb = lib.ggml_cpy(ctx, KQV_merged, lib.ggml_new_tensor_2d(ctx, lib.GGML_TYPE_F32, p.dim, p.seq_len))

        # projection (no bias)
        # final matmul to get the output of the attention
        s.xb2 = lib.ggml_mul_mat(ctx, s.xb, w.wo[l])

        # residual connection back into x
        s.x = lib.ggml_add(ctx, s.x, s.xb2)

        # ffn rmsnorm
        s.xb = lib.ggml_rms_norm(ctx, s.x, rms_norm_eps)
        s.xb = lib.ggml_mul(ctx, s.xb, w.rms_ffn_weight[l]) # xb = xb*ffn_norm(broadcasted)
        
        # Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        # first calculate self.w1(x) and self.w3(x)
        s.hb = lib.ggml_mul_mat(ctx, s.xb, w.w1[l])
        s.hb2 = lib.ggml_mul_mat(ctx, s.xb, w.w3[l])

        
        # SILU activation
        s.hb = lib.ggml_silu(ctx, s.hb)

        # elementwise multiply with w3(x)
        s.hb = lib.ggml_mul(ctx, s.hb, s.hb2)
        
        
        # final matmul to get the output of the ffn
        s.xb = lib.ggml_mul_mat(ctx, s.hb, w.w2[l]) # inverse?

        # residual connection
        s.x = lib.ggml_add(ctx, s.x, s.xb)
        
    # final rmsnorm
    s.x = lib.ggml_rms_norm(ctx, s.x, rms_norm_eps)
    s.x = lib.ggml_mul(ctx, s.x, w.rms_final_weight) # x = x*norm(broadcasted)

    # classifier into logits
    s.x = lib.ggml_mul_mat(ctx, w.wcls, s.x)

    lib.ggml_build_forward_expand(gf, s.x)


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
    copy(np.frombuffer(f.read(nbytes), dtype=np.float32).reshape(shape), tensor)

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
    if not w.shared_weights:
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
        # print('FINISHED READING Vocabulary')

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

    config = Config(*read_format(mm, '<iiiiiii'))
    shared_weights = config.vocab_size > 0
    config.vocab_size = int(math.fabs(config.vocab_size))

    print(config)

    ctx = init(mem_size=1024*1024*1024)

    # type = lib.GGML_TYPE_Q5_K
    tensor_type = lib.GGML_TYPE_F32

    weights = TransformerWeights(config, ctx, tensor_type, shared_weights)
    checkpoint_init_weights(mm, config, weights)
    
    mm.close()
    os.close(fd)

    vocab = read_vocab(tokenizer_model, config)
    state = RunState(config, ctx, tensor_type)

    # process the prompt, if any
    prompt_tokens = None
    if prompt is not None:
        prompt_tokens = bpe_encode(prompt, vocab)
        steps += len(prompt_tokens)

    print('prompt_tokens', prompt_tokens)

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
        token_str = vocab.vocab[next]+1 if (token == 1 and vocab.vocab[next][0] == ' ') else vocab.vocab[next]
        sys.stdout.write(token_str)
        sys.stdout.flush()
        lib.ggml_set_i32(token, next)

        # init the timer here because the first iteration can be slower
        # if (start == 0): start = time_in_ms(); }

    print("")

if __name__ == '__main__':
    CLI(run)
    