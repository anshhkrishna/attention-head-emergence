"""Attention-only transformer: no MLP blocks and no normalization layers, matching
the toy architecture the induction-head literature uses to isolate the attention
composition argument. Each layer is a single causal self-attention head reading from
and writing back into a shared residual stream. Forward and backward passes are
hand-derived and hand-written in NumPy; nothing here depends on an autograd framework.
"""
import numpy as np

_CAUSAL_MASK_CACHE = {}


def _causal_mask(seq_len):
    if seq_len not in _CAUSAL_MASK_CACHE:
        _CAUSAL_MASK_CACHE[seq_len] = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    return _CAUSAL_MASK_CACHE[seq_len]


def init_params(n_layers, d_model, vocab_size, seq_len, seed):
    """Random parameters for an n_layers-layer attention-only transformer. Every
    weight matrix is initialized independently from N(0, 1/d_model) -- no biases, no
    layer norm, matching the architecture's no-MLP, interpretability-first design.
    """
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(d_model)

    def mat(rows, cols):
        return rng.normal(0.0, scale, size=(rows, cols))

    params = {
        "W_E": mat(vocab_size, d_model),
        "W_pos": mat(seq_len, d_model),
        "W_U": mat(d_model, vocab_size),
        "layers": [
            {
                "W_Q": mat(d_model, d_model),
                "W_K": mat(d_model, d_model),
                "W_V": mat(d_model, d_model),
                "W_O": mat(d_model, d_model),
            }
            for _ in range(n_layers)
        ],
    }
    return params


def forward(params, tokens, cache=False):
    """Run the model on a (batch, seq_len) array of token ids. Returns logits of
    shape (batch, seq_len, vocab_size), predicting the token at position t+1 from
    positions <= t. With cache=True, also returns the intermediate activations
    `backward` needs.
    """
    batch_size, seq_len = tokens.shape
    d_model = params["W_E"].shape[1]
    mask = _causal_mask(seq_len)

    X = params["W_E"][tokens] + params["W_pos"][None, :seq_len, :]
    layer_cache = []
    for layer in params["layers"]:
        Q = X @ layer["W_Q"]
        K = X @ layer["W_K"]
        V = X @ layer["W_V"]
        scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_model)
        scores = np.where(mask, -np.inf, scores)
        shifted = scores - scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(shifted)
        A = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        Z = A @ V
        O = Z @ layer["W_O"]
        if cache:
            layer_cache.append({"X": X, "Q": Q, "K": K, "V": V, "A": A, "Z": Z})
        X = X + O

    logits = X @ params["W_U"]
    if cache:
        return logits, {"X_final": X, "layers": layer_cache}
    return logits


def cross_entropy_loss(logits, tokens):
    """Mean next-token cross-entropy: logits[:, t, :] predicts tokens[:, t+1], for
    every t in range(seq_len - 1). The last position has no target and is excluded.
    """
    pred_logits = logits[:, :-1, :]
    targets = tokens[:, 1:]
    shifted = pred_logits - pred_logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    b, s, _ = pred_logits.shape
    nll = -log_probs[np.arange(b)[:, None], np.arange(s)[None, :], targets]
    return nll.mean()


def backward(params, tokens, cache):
    """Gradient of cross_entropy_loss(forward(params, tokens, cache=True), tokens)
    with respect to every array in `params`, returned in the same nested structure.
    Derived by hand: softmax-cross-entropy backward, then per layer, softmax
    (attention) backward, then the linear Q/K/V/O projections, accumulated through
    the residual stream.
    """
    batch_size, seq_len = tokens.shape
    d_model = params["W_E"].shape[1]
    mask = _causal_mask(seq_len)

    X_final = cache["X_final"]
    logits = X_final @ params["W_U"]
    pred_logits = logits[:, :-1, :]
    targets = tokens[:, 1:]
    b, s, v = pred_logits.shape

    shifted = pred_logits - pred_logits.max(axis=-1, keepdims=True)
    exp_shift = np.exp(shifted)
    probs = exp_shift / exp_shift.sum(axis=-1, keepdims=True)
    onehot = np.zeros_like(probs)
    onehot[np.arange(b)[:, None], np.arange(s)[None, :], targets] = 1.0
    dpred_logits = (probs - onehot) / (b * s)

    dlogits = np.zeros_like(logits)
    dlogits[:, :-1, :] = dpred_logits

    grads = {"W_U": np.einsum("bsd,bsv->dv", X_final, dlogits)}
    dX = dlogits @ params["W_U"].T

    grads["layers"] = [None] * len(params["layers"])
    for l in reversed(range(len(params["layers"]))):
        layer = params["layers"][l]
        lc = cache["layers"][l]

        dO = dX
        dWO = np.einsum("bsd,bse->de", lc["Z"], dO)
        dZ = dO @ layer["W_O"].T

        dA = np.einsum("bsd,btd->bst", dZ, lc["V"])
        dV = np.einsum("bst,bsd->btd", lc["A"], dZ)

        sum_term = np.sum(lc["A"] * dA, axis=-1, keepdims=True)
        dscores = lc["A"] * (dA - sum_term)
        dscores = np.where(mask, 0.0, dscores)
        dscores = dscores / np.sqrt(d_model)

        dQ = dscores @ lc["K"]
        dK = np.einsum("bst,bsd->btd", dscores, lc["Q"])

        dWQ = np.einsum("bsd,bse->de", lc["X"], dQ)
        dWK = np.einsum("bsd,bse->de", lc["X"], dK)
        dWV = np.einsum("bsd,bse->de", lc["X"], dV)

        dX_attn = dQ @ layer["W_Q"].T + dK @ layer["W_K"].T + dV @ layer["W_V"].T
        dX = dX + dX_attn

        grads["layers"][l] = {"W_Q": dWQ, "W_K": dWK, "W_V": dWV, "W_O": dWO}

    grads["W_pos"] = np.zeros_like(params["W_pos"])
    grads["W_pos"][:seq_len] = dX.sum(axis=0)
    grads["W_E"] = np.zeros_like(params["W_E"])
    np.add.at(grads["W_E"], tokens, dX)

    return grads
