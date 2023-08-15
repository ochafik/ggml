# Ported from llama2.c
# https://github.com/karpathy/llama2.c/blob/master/run.c
# Use faster tokenization, and naively allocate buffers in GGML / wrap them to numpy

# cmake ../../../llama.cpp -B /tmp/llama_release -DLLAMA_METAL=1 -DBUILD_SHARED_LIBS=1 -DLLAMA_BUILD_EXAMPLES=0 -DLLAMA_BUILD_TESTS=0 -DCMAKE_BUILD_TYPE=Release && ( cd /tmp/llama_release && make -j )
# import os; os.environ['GGML_LIBRARY'] = '/tmp/llama_release/libggml_shared.dylib'

# cmake ../../../llama.cpp -B /tmp/llama_debug -DLLAMA_METAL=1 -DBUILD_SHARED_LIBS=1 -DLLAMA_BUILD_EXAMPLES=0 -DLLAMA_BUILD_TESTS=0 -DCMAKE_BUILD_TYPE=Debug && ( cd /tmp/llama_debug && make -j )
import os; os.environ['GGML_LIBRARY'] = '/tmp/llama_debug/libggml_shared.dylib'

# python llama2.c.py ~/AI/Models/llama2.c.stories15M.bin ../../../llama2.c/tokenizer.bin --prompt "Hello, world"

from ggml import lib, ffi
from ggml.utils import init, numpy, copy
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict, Any, Callable, TypeVar, Generic
from jsonargparse import CLI
from pathlib import Path
import struct, os, mmap, math, sys, time, inspect
from collections import namedtuple
import scipy as sp

Tensor = ffi.CData
Context = ffi.CData
TensorLike = Union[Tensor, np.ndarray]

__tensor_factories = {
    1: lib.ggml_new_tensor_1d,
    2: lib.ggml_new_tensor_2d,
    3: lib.ggml_new_tensor_3d,
    4: lib.ggml_new_tensor_4d,
}

def _get_shape(x: TensorLike): return x.shape if isinstance(x, np.ndarray) else tuple([x.ne[i] for i in range(x.n_dims)])

def _new_tensor(ctx, shape, type):
    factory = __tensor_factories[len(shape)]
    if factory:
        return factory(ctx, type, *shape)

    dims = ffi.new('int[]', len(shape))
    for i, dim in enumerate(shape): dims[i] = dim
    return lib.ggml_new_tensor(ctx, type, len(shape), dims)

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
        self.token_embedding_table = _new_tensor(ctx, (p.vocab_size, p.dim), type)
        # weights for rmsnorms
        self.rms_att_weight = [_new_tensor(ctx, (p.dim,), type) for _ in range(p.n_layers)]
        self.rms_ffn_weight = [_new_tensor(ctx, (p.dim,), type) for _ in range(p.n_layers)]
        # weights for matmuls
        self.wq = [_new_tensor(ctx, (p.dim, p.dim), type) for _ in range(p.n_layers)]
        self.wk = [_new_tensor(ctx, (p.dim, p.dim), type) for _ in range(p.n_layers)]
        self.wv = [_new_tensor(ctx, (p.dim, p.dim), type) for _ in range(p.n_layers)]
        self.wo = [_new_tensor(ctx, (p.dim, p.dim), type) for _ in range(p.n_layers)]
        # weights for ffn
        self.w1 = [_new_tensor(ctx, (p.dim, p.hidden_dim), type) for _ in range(p.n_layers)]
        self.w2 = [_new_tensor(ctx, (p.hidden_dim, p.dim), type) for _ in range(p.n_layers)]
        self.w3 = [_new_tensor(ctx, (p.dim, p.hidden_dim), type) for _ in range(p.n_layers)]
        # final rmsnorm
        self.rms_final_weight = _new_tensor(ctx, (p.dim,), type)
        # freq_cis for RoPE relatively positional embeddings
        self.freq_cis_real = _new_tensor(ctx, (p.seq_len, p.head_size // 2), type)
        self.freq_cis_imag = _new_tensor(ctx, (p.seq_len, p.head_size // 2), type)
        # (optional) classifier weights for the logits, on the last layer
        self.wcls = _new_tensor(ctx, (p.dim, p.vocab_size), type)
        self.shared_weights = shared_weights

        # numpy array views of each tensor
        self.token_embedding_table_ = numpy(self.token_embedding_table)
        self.freq_cis_real_ = numpy(self.freq_cis_real)
        self.freq_cis_imag_ = numpy(self.freq_cis_imag)

# struct used when sorting probabilities during top-p sampling
@dataclass
class ProbIndex:
    prob: float
    index: int

# current wave of activations
@dataclass
class RunState:
    def __init__(self, config: Config, ctx: int, tensor_type: int):
        # activation at current time stamp (dim,)
        self.x = _new_tensor(ctx, (config.dim,), tensor_type)
        # same, but inside a residual branch (dim,)
        self.xb = _new_tensor(ctx, (config.dim,), tensor_type)
         # an additional buffer just for convenience (dim,)
        self.xb2 = _new_tensor(ctx, (config.dim,), tensor_type)
        # buffer for scores/attention values
        self.att = _new_tensor(ctx, (config.n_heads, config.seq_len), tensor_type)
        # kv cache
        self.key_cache = _new_tensor(ctx, (config.n_layers, config.seq_len, config.dim), tensor_type)
        self.value_cache = _new_tensor(ctx, (config.n_layers, config.seq_len, config.dim), tensor_type)

        # numpy array views of each tensor
        self.x_ = numpy(self.x)
        self.xb_ = numpy(self.xb)
        self.xb_heads_ = self.xb_.reshape(config.n_heads, config.head_size)
        self.xb2_ = numpy(self.xb2)
        self.att_ = numpy(self.att)
        self.key_cache_ = numpy(self.key_cache)
        self.value_cache_ = numpy(self.value_cache)
        # Split these by heads for convenience
        self.key_cache_heads_ = self.key_cache_.reshape(config.n_layers, config.seq_len, config.n_heads, config.head_size)
        self.value_cache_heads_ = self.value_cache_.reshape(config.n_layers, config.seq_len, config.n_heads, config.head_size)

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

def get_np(a: TensorLike) -> np.ndarray:
    if isinstance(a, np.ndarray): return a
    return numpy(a)

@dataclass(frozen=True)
class MatrixShape:
    type: int
    shape: tuple[int]
    transpose: bool

class Context:
    def __init__(self, ctx: ffi.CData, n_threads: int):
        self.ctx = ctx
        # self.ctx_metal = lib.ggml_metal_init(1)
        # ggml_metal_add_buffer(ctx->ctx_metal, "data", data_ptr, data_size, max_size)
        self.cache = {}
        self.n_threads = n_threads

    def _create_graph(self, input_shapes, intermediates, out):
        vars = {
            name: _new_tensor(self.ctx, input.shape, input.type)
            for name, input in input_shapes.items()
        }
        
        def invoke(f):
            argnames = inspect.getfullargspec(f).args
            args = [self.ctx if n == 'ctx' else vars[n] for n in argnames]
            return f(*args)
        
        for name, f in intermediates.items():
            vars[name] = invoke(f)

        outputs = invoke(out)

        gf = lib.ggml_new_graph(self.ctx)
        for output in (outputs if isinstance(outputs, tuple) else [outputs]):
            lib.ggml_build_forward_expand(gf, output)

        return (
            gf,
            vars,
            outputs,
        )

    def execute(self, name, inputs, intermediates, out, n_threads=None, cache=True):
        input_shapes = {
            name: MatrixShape(input.type, _get_shape(input), transpose=False)
            for name, input in inputs.items()
        }
        def make_graph():
            return self._create_graph(input_shapes, intermediates, out)
        (g, vars, outs) = self._get_cache((name, tuple(input_shapes)), make_graph) \
            if cache else make_graph()

        for name, input in inputs.items():
            ffi.memmove(vars[name].data, input.data, lib.ggml_nbytes(input))

        lib.ggml_graph_compute_with_ctx(self.ctx, g, n_threads or self.n_threads)
        return outs

    def _get_cache(self, key, factory: Callable[[], Any]):
        val = self.cache.get(key)
        if val is None:
            val = factory()
            self.cache[key] = val
        return val

rms_norm_eps = 1e-5
rope_freq_base  = 10000.0
rope_freq_scale = 1.0
       
def transformer(ctx: Context, token: int, pos: int, p: Config, s: RunState, w: TransformerWeights):
    dtype = np.float32

    # a few convenience variables
    x = s.x
    x_ = s.x_
    dim = p.dim
    head_size = dim // p.n_heads
    head_size_sqrt = math.sqrt(head_size)
    assert dim % p.n_heads == 0

    # copy the token embedding into x
    content_row_ = w.token_embedding_table_[token, :]
    assert content_row_.shape == (dim,)
    np.copyto(x_, content_row_)

    # pluck out the "pos" row of freq_cis_real and freq_cis_imag
    freq_cis_real_row = w.freq_cis_real_[pos]
    freq_cis_imag_row = w.freq_cis_imag_[pos]

    zero_xb = np.zeros(dim, dtype)

    def rmsnorm(ctx, x, w):
        return lib.ggml_mul(ctx, lib.ggml_rms_norm(ctx, x, rms_norm_eps), w)

    def mulmat(ctx, a, b):
        if a.n_dims == 2 and a.ne[0] == 1 and a.ne[0] != b.ne[0]:
            a = lib.ggml_reshape_1d(ctx, a, a.ne[1])
        return lib.ggml_mul_mat(ctx, a, b)
    
    def add(ctx, a, b):
        if b.n_dims == 2 and b.ne[0] == 1:
            b = lib.ggml_reshape_1d(ctx, b, b.ne[1])
        return lib.ggml_add(ctx, a, b)
    
    def mul(ctx, a, b):
        return lib.ggml_mul(ctx, a, b)
    
    def rope(ctx, a, pos):
        return lib.ggml_reshape_1d(
            ctx,
            # Broadcast rope encoding to all heads
            lib.ggml_rope_custom_inplace(
                ctx,
                lib.ggml_reshape_2d(ctx, a, p.head_size, p.n_kv_heads),
                pos, p.head_size, 0, 0, rope_freq_base, rope_freq_scale),
            p.dim)

    # forward all the layers
    for l in range(p.n_layers):
        
        # Fused graph for attention rmsnorm, qkv matmuls, RoPE encoding of q & k for this position
        (q, k, v) = ctx.execute(
            "rmsnorm + qkv matmuls",
            {
                "x": x,
                "att_w": w.rms_att_weight[l],
                "wq": w.wq[l],
                "wk": w.wk[l],
                "wv": w.wv[l]
            }, {
                # attention rmsnorm
                "normout": lambda ctx, x, att_w: rmsnorm(ctx, x, att_w),
            },
            # qkv matmuls for this position and RoPE for q & k
            lambda ctx, normout, wq, wk, wv: (
                rope(ctx, mulmat(ctx, normout, wq), pos), # per-head RoPE(q = normout * wq)
                rope(ctx, mulmat(ctx, normout, wk), pos), # per-head RoPE(k = normout * wk)
                mulmat(ctx, normout, wv), # v = normout * wv
            ))

        # save key,value at this time step (pos) to our kv cache
        copy(k, s.key_cache_[l, pos])
        copy(v, s.value_cache_[l, pos])

        copy(zero_xb, s.xb_)

        q_ = numpy(q).reshape((dim,))
        q_heads_ = q_.reshape(p.n_heads, p.head_size)
        
        # multihead attention. iterate over all heads
        for h in range(p.n_heads):
            # get the query vector for this head
            q = q_heads_[h]
            # attention scores for this head
            att = s.att_[h, :]
            # iterate over all timesteps, including the current one
            for t in range(pos+1):
                # get the key vector for this head and at this timestep
                k = s.key_cache_heads_[l, t, h]
                # calculate the attention score as the dot product of q and k
                score = np.dot(q, k) / head_size_sqrt
                # save the score to the attention buffer
                att[t] = score

            # softmax the scores to get attention weights, from 0..pos inclusively
            att[0:pos+1] = sp.special.softmax(att[0:pos+1])

            # weighted sum of the values, store back into xb

            for t in range(pos+1):
                # get the value vector for this head and at this timestep
                v = s.value_cache_heads_[l, t, h]
                # get the attention weight for this timestep
                a = att[t]
                # accumulate the weighted value into xb
                for i in range(head_size):
                    s.xb_heads_[h, i] += a * v[i]

        x = ctx.execute(
            "attention",
            {   
                "x": x,
                "xb": s.xb,
                "ffn_w": w.rms_ffn_weight[l],
                "w1": w.w1[l], "w2": w.w2[l], "w3": w.w3[l], "wo": w.wo[l] },
            {
                "xx": lambda ctx, x, xb, wo:
                    # residual connection back into x
                    add(
                        ctx, x,
                        # projection (no bias)
                        # final matmul to get the output of the attention   
                        mulmat(ctx, xb, wo)
                    ),
                "normout": lambda ctx, xx, ffn_w:
                    # ffn rmsnorm
                    rmsnorm(ctx, xx, ffn_w)
            },
            lambda ctx, normout, xx, w1, w2, w3:
                # residual connection
                add(ctx, xx, 
                    # final matmul by w2 to get the output of the ffn
                    mulmat(
                        ctx,
                        # silu(w1(x)) x w3(x)
                        mul(
                            ctx,
                            lib.ggml_silu(ctx, mulmat(ctx, normout, w1)),
                            mulmat(ctx, normout, w3)),
                        w2)))

    return ctx.execute(
        "final rms + logits",
        {   
            "x": x,
            "final_w": w.rms_final_weight,
            "wcls": w.wcls,
        },
        {},
        lambda ctx, final_w, wcls:
            # classifier into logits
            mulmat(ctx,
                # final rmsnorm
                rmsnorm(ctx, x, final_w),
                w.wcls))

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

def read_tensor(name: str, f, tensor: ffi.CData, permute=False):
    shape = tuple([tensor.ne[i] for i in range(tensor.n_dims)])
    nbytes = np.prod(shape) * ffi.sizeof('float')
    array = np.frombuffer(f.read(nbytes), dtype=np.float32).reshape(shape)
    # if permute: array = np.ascontiguousarray(np.transpose(array)).reshape(shape)
    copy(array, tensor)

def checkpoint_init_weights(mm, p: Config, w: TransformerWeights):
    read_tensor("token_embedding_table", mm, w.token_embedding_table)
    for i in range(p.n_layers): read_tensor(f'{w.rms_att_weight[i]}', mm, w.rms_att_weight[i])
    for i in range(p.n_layers): read_tensor(f'{w.wq[i]}', mm, w.wq[i], permute=True)
    for i in range(p.n_layers): read_tensor(f'{w.wk[i]}', mm, w.wk[i], permute=True)
    for i in range(p.n_layers): read_tensor(f'{w.wv[i]}', mm, w.wv[i], permute=True)
    for i in range(p.n_layers): read_tensor(f'{w.wo[i]}', mm, w.wo[i], permute=True)
    for i in range(p.n_layers): read_tensor(f'{w.rms_ffn_weight[i]}', mm, w.rms_ffn_weight[i], permute=True)
    for i in range(p.n_layers): read_tensor(f'{w.w1[i]}', mm, w.w1[i], permute=True)
    for i in range(p.n_layers): read_tensor(f'{w.w2[i]}', mm, w.w2[i], permute=True)
    for i in range(p.n_layers): read_tensor(f'{w.w3[i]}', mm, w.w3[i], permute=True)
    read_tensor('rms_final_weight', mm, w.rms_final_weight)
    read_tensor('freq_cis_real', mm, w.freq_cis_real)
    read_tensor('freq_cis_imag', mm, w.freq_cis_imag)
    if w.shared_weights:
        copy(np.ascontiguousarray(w.token_embedding_table_.transpose()), w.wcls_)
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

def sample(probabilities: TensorLike) -> int:
    probabilities = get_np(probabilities)
    (n,) = probabilities.shape
    # sample index from probabilities (they must sum to 1!)
    r = np.random.rand()
    cdf = 0.0
    for i in range(n):
        cdf += probabilities[i]
        if r < cdf:
            return i
    return n - 1 # in case of rounding errors

def sample_topp(probabilities: TensorLike, n: int, topp: float) -> int:
    # top-p sampling (or "nucleus sampling") samples from the smallest set of
    # tokens that exceed probability topp. This way we never sample tokens that
    # have very low probabilities and are less likely to go "off the rails".

    # quicksort indices in descending order of probabilities
    probindex = [ProbIndex(probabilities[i], i) for i in range(n)]
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

def run(
        model: Path,
        tokenizer_model: Path,
        prompt: Optional[str] = None,
        steps: int = 128,
        # temperature: float = 0.0,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        n_threads: int = 8,
        topp: float = 0.9): # top-p in nucleus sampling

    np.random.seed(seed)

    fd = os.open(model.as_posix(), os.O_RDONLY)
    mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)

    config = Config(*read_format(mm, '<iiiiiii'))
    shared_weights = config.vocab_size > 0
    config.vocab_size = int(math.fabs(config.vocab_size))

    print(config)

    ctx = init(mem_size=1024*1024*1024)
    context = Context(ctx, n_threads)

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
    start = 0   # used to time our code, only initialized after first iteration
    next = None # will store the next token in the sequence
    token = 1   # init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
    pos = 0     # position in the sequence
    
    while (pos < steps):

        # forward the transformer to get logits for the next token
        logits = numpy(transformer(context, token, pos, config, state, weights)).transpose()

        # advance the state state machine
        if(pos < (len(prompt_tokens) if prompt_tokens else 0)):
            # if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos]
        else:
            # sample the next token
            if (temperature == 0.0):
                # greedy argmax sampling: take the token with the highest probability
                next = np.argmax(logits)
            else:
                # apply the temperature to the logits
                for q in range(config.vocab_size): logits[q] /= temperature
                # apply softmax to the logits to get the probabilities for next token
                logits = sp.special.softmax(logits)
                # we sample from this distribution to get the next token
                if (topp <= 0):
                    # simply sample from the predicted probability distribution
                    next = sample(logits, config.vocab_size)
                else:
                    # top-p (nucleus) sampling, clamping the least likely tokens to zero
                    next = sample_topp(logits, config.vocab_size, topp)

        pos += 1

        # data-dependent terminating condition: the BOS (1) token delimits sequences
        if (next == 1): break

        # following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
        if token == 1 and vocab.vocab[next][0] == ' ':
            token_str = vocab.vocab[next][1:]
        else:
            token_str = vocab.vocab[next]
        sys.stdout.write(token_str)
        sys.stdout.flush()
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
    