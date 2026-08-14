"""Two reference points for the induction task: a 2-layer model at random
initialization, which has never seen data, and a 1-layer model trained on the same
task and the same number of steps the eventual 2-layer method run will use. Both are
expected to sit near chance on induction positions -- the random-init model because
weights are untrained, the trained 1-layer model because a single attention layer
cannot implement the previous-token-head -> induction-head composition the task
requires (Elhage et al., 2021).
"""
import numpy as np

from data import EVAL_SEED, HALF_LEN, SEQ_LEN, VOCAB_SIZE, make_batch
from model import backward, cross_entropy_loss, forward, init_params

D_MODEL = 24
MODEL_SEED = 100
TRAIN_RNG_SEED = 200
NUM_TRAIN_STEPS = 500
BATCH_SIZE = 64
LEARNING_RATE = 1e-2
EVAL_BATCH_SIZE = 1024

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


def train_baseline(n_layers, num_steps, seed_model, seed_train, log_every=100):
    """Train an n_layers-layer model with Adam on fresh synthetic batches drawn from
    a training RNG seeded independently of both REFERENCE_SEED and EVAL_SEED, so
    evaluation data is never seen during training.
    """
    params = init_params(n_layers, D_MODEL, VOCAB_SIZE, SEQ_LEN, seed=seed_model)
    m = _zeros_like_params(params)
    v = _zeros_like_params(params)
    t = 0
    rng = np.random.default_rng(seed_train)
    for step in range(1, num_steps + 1):
        batch_seed = int(rng.integers(0, 2**31 - 1))
        tokens, _ = make_batch(BATCH_SIZE, seed=batch_seed)
        logits, cache = forward(params, tokens, cache=True)
        loss = cross_entropy_loss(logits, tokens)
        grads = backward(params, tokens, cache)
        t = _adam_step(params, grads, m, v, t, LEARNING_RATE)
        if step == 1 or step % log_every == 0:
            print(f"    step {step:4d}/{num_steps}  loss={loss:.4f}")
    return params


def evaluate(params, seed=EVAL_SEED, batch_size=EVAL_BATCH_SIZE):
    """Accuracy on a held-out batch, broken into three groups of positions:

    - non-induction: no earlier occurrence of the current token exists; chance by
      construction (1 / VOCAB_SIZE), and a sanity check on the metric plumbing itself.
    - fixed-offset induction: positions in the second half of the sequence, where the
      correct next token is always exactly HALF_LEN positions back. A model can solve
      these with pure position-based attention (attend HALF_LEN back, unconditional on
      content) -- no content matching and no induction mechanism required.
    - coincidental induction: positions where the current token happens to repeat an
      earlier one at no fixed offset (chance collisions inside the random prefix).
      There is no positional shortcut here: solving these requires attending to
      wherever the matching content occurred, which is what "induction" means.

    Lumping the last two together (as a plain "induction accuracy") over-credits any
    model that has merely learned the fixed HALF_LEN offset, so both the lumped number
    and the split are returned.
    """
    tokens, mask = make_batch(batch_size, seed=seed)
    logits = forward(params, tokens, cache=False)
    preds = logits[:, :-1, :].argmax(axis=-1)
    targets = tokens[:, 1:]
    correct = preds == targets
    mask_valid = mask[:, :-1]

    position = np.arange(SEQ_LEN - 1)[None, :]
    fixed_offset_mask = mask_valid & (position >= HALF_LEN)
    coincidental_mask = mask_valid & (position < HALF_LEN)

    return {
        "induction_lumped": correct[mask_valid].mean(),
        "induction_fixed_offset": correct[fixed_offset_mask].mean(),
        "induction_coincidental": correct[coincidental_mask].mean(),
        "non_induction": correct[~mask_valid].mean(),
    }


def _print_eval(metrics, chance):
    print(f"  induction accuracy (lumped):        {metrics['induction_lumped']:.4f}")
    print(f"  induction accuracy (fixed-offset):  {metrics['induction_fixed_offset']:.4f}"
          "  (solvable by pure position-based attention, no content matching)")
    print(f"  induction accuracy (coincidental):  {metrics['induction_coincidental']:.4f}"
          "  (no positional shortcut -- the actual induction test)")
    print(f"  non-induction accuracy:             {metrics['non_induction']:.4f}  (expect ~{chance:.4f})")


def main():
    chance = 1.0 / VOCAB_SIZE
    print(f"chance accuracy: 1/{VOCAB_SIZE} = {chance:.4f}")
    print(f"eval batch: {EVAL_BATCH_SIZE} sequences, seed={EVAL_SEED}")
    print(
        "note: the second half of every sequence is an exact repeat of the first, so"
        " 'induction accuracy' as usually defined lumps together positions solvable by"
        " a fixed HALF_LEN positional offset (no content matching needed) with positions"
        " that repeat by chance at no fixed offset (the actual test of content-based"
        " induction). Both are reported separately below.\n"
    )

    print("baseline A: 2-layer model at random initialization (untrained)")
    params_random = init_params(2, D_MODEL, VOCAB_SIZE, SEQ_LEN, seed=MODEL_SEED)
    _print_eval(evaluate(params_random), chance)
    print()

    print(f"baseline B: 1-layer model, trained {NUM_TRAIN_STEPS} steps, batch {BATCH_SIZE}")
    print(f"  model_seed={MODEL_SEED} train_rng_seed={TRAIN_RNG_SEED} lr={LEARNING_RATE}")
    params_trained = train_baseline(1, NUM_TRAIN_STEPS, MODEL_SEED, TRAIN_RNG_SEED)
    _print_eval(evaluate(params_trained), chance)


if __name__ == "__main__":
    main()
