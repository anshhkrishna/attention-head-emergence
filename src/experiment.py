"""Full training run of the 2-layer attention-only transformer on the induction task.
Checkpoints induction/non-induction accuracy on the held-out eval batch and the raw
per-head attention matrices for a fixed probe sequence at regular step intervals, so a
later stage can render both the accuracy trajectory and the attention pattern that
produced it.
"""
import time

import numpy as np

from baselines import evaluate
from data import EVAL_SEED, VOCAB_SIZE, make_batch
from model import forward, init_params
from train import BATCH_SIZE, D_MODEL, LEARNING_RATE, NUM_TRAIN_STEPS, train_model

N_LAYERS = 2
MODEL_SEED = 100
TRAIN_RNG_SEED = 201
CHECKPOINT_EVERY = 20
PROBE_SEED = 2


def _probe_sequence():
    tokens, _ = make_batch(1, seed=PROBE_SEED)
    return tokens


def _attention_snapshot(params, probe_tokens):
    """Per-layer attention matrices for the probe sequence: (n_layers, seq, seq)."""
    _, cache = forward(params, probe_tokens, cache=True)
    return np.stack([layer["A"][0] for layer in cache["layers"]])


def main():
    chance = 1.0 / VOCAB_SIZE
    probe_tokens = _probe_sequence()

    steps_record = []
    attn_record = []

    def _log_checkpoint(step, params):
        m = evaluate(params, seed=EVAL_SEED)
        attn_record.append(_attention_snapshot(params, probe_tokens))
        steps_record.append(step)
        print(
            f"step {step:4d}/{NUM_TRAIN_STEPS}  "
            f"induction(coincidental)={m['induction_coincidental']:.4f}  "
            f"induction(fixed_offset)={m['induction_fixed_offset']:.4f}  "
            f"non_induction={m['non_induction']:.4f}  (chance={chance:.4f})"
        )

    print(f"chance accuracy: 1/{VOCAB_SIZE} = {chance:.4f}")
    print(
        f"2-layer model: model_seed={MODEL_SEED} train_rng_seed={TRAIN_RNG_SEED} "
        f"num_steps={NUM_TRAIN_STEPS} batch_size={BATCH_SIZE} lr={LEARNING_RATE}"
    )
    print(f"checkpoint every {CHECKPOINT_EVERY} steps, probe_seed={PROBE_SEED}\n")

    initial_params = init_params(
        N_LAYERS, D_MODEL, VOCAB_SIZE, probe_tokens.shape[1], seed=MODEL_SEED
    )
    _log_checkpoint(0, initial_params)

    def callback(step, params):
        if step % CHECKPOINT_EVERY == 0:
            _log_checkpoint(step, params)

    start = time.time()
    final_params, losses = train_model(
        N_LAYERS, NUM_TRAIN_STEPS, MODEL_SEED, TRAIN_RNG_SEED, callback=callback
    )
    elapsed = time.time() - start

    if NUM_TRAIN_STEPS % CHECKPOINT_EVERY != 0:
        _log_checkpoint(NUM_TRAIN_STEPS, final_params)

    assert all(np.isfinite(l) for l in losses), "loss went non-finite during training"
    print(f"\nwall time: {elapsed:.1f}s for {NUM_TRAIN_STEPS} steps")

    np.savez(
        "results/attention_checkpoints.npz",
        steps=np.array(steps_record),
        attention=np.stack(attn_record),
        probe_tokens=probe_tokens[0],
    )
    print(f"saved {len(steps_record)} checkpoints to results/attention_checkpoints.npz")


if __name__ == "__main__":
    main()
