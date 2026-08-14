"""Reusable Adam training loop for the attention-only transformer, generic over
layer count so the 2-layer method run and any future ablation share one
implementation.
"""
import numpy as np

from data import SEQ_LEN, VOCAB_SIZE, make_batch
from model import backward, cross_entropy_loss, forward, init_params

D_MODEL = 24
BATCH_SIZE = 64
LEARNING_RATE = 1e-2
NUM_TRAIN_STEPS = 500

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8


def _zeros_like_params(params):
    return {
        "W_E": np.zeros_like(params["W_E"]),
        "W_pos": np.zeros_like(params["W_pos"]),
        "W_U": np.zeros_like(params["W_U"]),
        "layers": [{k: np.zeros_like(v) for k, v in layer.items()} for layer in params["layers"]],
    }


def _adam_update(arr, g, m, v, t, lr):
    m[:] = ADAM_BETA1 * m + (1 - ADAM_BETA1) * g
    v[:] = ADAM_BETA2 * v + (1 - ADAM_BETA2) * (g * g)
    m_hat = m / (1 - ADAM_BETA1**t)
    v_hat = v / (1 - ADAM_BETA2**t)
    arr -= lr * m_hat / (np.sqrt(v_hat) + ADAM_EPS)


def _adam_step(params, grads, m, v, t, lr):
    t = t + 1
    for key in ("W_E", "W_pos", "W_U"):
        _adam_update(params[key], grads[key], m[key], v[key], t, lr)
    for layer, gl, ml, vl in zip(params["layers"], grads["layers"], m["layers"], v["layers"]):
        for key in layer:
            _adam_update(layer[key], gl[key], ml[key], vl[key], t, lr)
    return t


def train_model(n_layers, num_steps, seed_model, seed_train, batch_size=BATCH_SIZE,
                 lr=LEARNING_RATE, log_every=100, callback=None):
    """Train an n_layers-layer model with Adam on fresh synthetic batches drawn from
    a training RNG independent of `REFERENCE_SEED`/`EVAL_SEED`, so evaluation data is
    never seen during training. Returns (params, losses): the trained parameters and
    the per-step loss history. If given, `callback(step, params)` runs after every
    step, letting a caller record checkpoints without this loop needing to know what
    the caller wants to record.
    """
    params = init_params(n_layers, D_MODEL, VOCAB_SIZE, SEQ_LEN, seed=seed_model)
    m = _zeros_like_params(params)
    v = _zeros_like_params(params)
    t = 0
    rng = np.random.default_rng(seed_train)
    losses = []
    for step in range(1, num_steps + 1):
        batch_seed = int(rng.integers(0, 2**31 - 1))
        tokens, _ = make_batch(batch_size, seed=batch_seed)
        logits, cache = forward(params, tokens, cache=True)
        loss = cross_entropy_loss(logits, tokens)
        grads = backward(params, tokens, cache)
        t = _adam_step(params, grads, m, v, t, lr)
        losses.append(float(loss))
        if step == 1 or step % log_every == 0:
            print(f"    step {step:4d}/{num_steps}  loss={loss:.4f}")
        if callback is not None:
            callback(step, params)
    return params, losses


def main():
    print("smoke test: 2-layer model on a tiny subset (256 sequences/step, 8 steps)")
    _, losses = train_model(n_layers=2, num_steps=8, seed_model=100, seed_train=201,
                             batch_size=256, log_every=1)
    assert all(np.isfinite(l) for l in losses), "loss went non-finite"
    print(f"loss: {losses[0]:.4f} -> {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
