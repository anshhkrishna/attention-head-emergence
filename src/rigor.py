"""Multi-seed rigor sweep for the 2-layer method run: repeats training across several
(model_seed, train_rng_seed) pairs to check whether the coincidental-induction result
from a single seed holds generally, records the training step at which each seed's
coincidental accuracy crosses the halfway point between chance and its own final
value, and verifies programmatically that no training batch ever coincides with the
held-out evaluation batch.
"""
import numpy as np

from baselines import NUM_TRAIN_STEPS as BASELINE_1LAYER_STEPS
from baselines import evaluate
from data import EVAL_SEED, VOCAB_SIZE, make_batch
from model import init_params
from train import BATCH_SIZE, D_MODEL, LEARNING_RATE, NUM_TRAIN_STEPS, train_model

N_LAYERS = 2
CHECKPOINT_EVERY = 20
SEEDS = [
    (110, 310),
    (111, 311),
    (112, 312),
    (113, 313),
    (114, 314),
]
BASELINE_1LAYER_COINCIDENTAL = 0.1777  # trained 1-layer baseline, results/baseline.log

assert NUM_TRAIN_STEPS == BASELINE_1LAYER_STEPS, (
    "rigor sweep must train for the same step count as the already-trained 1-layer "
    "baseline for the comparison to be fair"
)


def _training_batch_seeds(seed_train, num_steps):
    """Regenerate the exact sequence of per-step batch seeds train_model() draws for
    seed_train, without re-running training -- used to reconstruct the token batches
    actually seen during training for the leakage check below.
    """
    rng = np.random.default_rng(seed_train)
    return [int(rng.integers(0, 2**31 - 1)) for _ in range(num_steps)]


def check_no_leakage(seed_trains, num_steps, eval_seed=EVAL_SEED):
    """Confirms, by direct token-array comparison rather than by seed-value
    construction, that no batch drawn during training across any seed_train in
    seed_trains is identical to any row of the held-out evaluation batch.
    """
    eval_tokens, _ = make_batch(1024, seed=eval_seed)
    eval_rows = {row.tobytes() for row in eval_tokens}

    collisions = 0
    total_rows = 0
    for seed_train in seed_trains:
        for batch_seed in _training_batch_seeds(seed_train, num_steps):
            tokens, _ = make_batch(BATCH_SIZE, seed=batch_seed)
            total_rows += tokens.shape[0]
            for row in tokens:
                if row.tobytes() in eval_rows:
                    collisions += 1
    return collisions, total_rows


def _transition_step(steps, accuracies, chance):
    """First checkpoint step at which accuracy crosses halfway between chance and the
    run's own final accuracy. Returns None if the final accuracy does not clear
    chance, since "halfway to chance" is not a meaningful threshold in that case.
    """
    final = accuracies[-1]
    if final <= chance:
        return None
    threshold = chance + 0.5 * (final - chance)
    for step, acc in zip(steps, accuracies):
        if acc >= threshold:
            return step
    return None


def run_seed(model_seed, train_rng_seed):
    steps_record = []
    coincidental_record = []
    fixed_offset_record = []

    def _checkpoint(step, params):
        m = evaluate(params, seed=EVAL_SEED)
        steps_record.append(step)
        coincidental_record.append(m["induction_coincidental"])
        fixed_offset_record.append(m["induction_fixed_offset"])
        print(
            f"    step {step:4d}/{NUM_TRAIN_STEPS}  "
            f"induction(coincidental)={m['induction_coincidental']:.4f}  "
            f"induction(fixed_offset)={m['induction_fixed_offset']:.4f}"
        )

    def callback(step, params):
        if step % CHECKPOINT_EVERY == 0:
            _checkpoint(step, params)

    initial_params = init_params(N_LAYERS, D_MODEL, VOCAB_SIZE,
                                  make_batch(1, seed=EVAL_SEED)[0].shape[1], seed=model_seed)
    _checkpoint(0, initial_params)
    train_model(N_LAYERS, NUM_TRAIN_STEPS, model_seed, train_rng_seed,
                batch_size=BATCH_SIZE, lr=LEARNING_RATE, callback=callback)

    return steps_record, coincidental_record, fixed_offset_record


def main():
    chance = 1.0 / VOCAB_SIZE
    print(f"chance accuracy: 1/{VOCAB_SIZE} = {chance:.4f}")
    print(f"seeds: {SEEDS}")
    print(f"num_train_steps={NUM_TRAIN_STEPS} batch_size={BATCH_SIZE} lr={LEARNING_RATE}\n")

    final_coincidental = []
    transition_steps = []

    for model_seed, train_rng_seed in SEEDS:
        print(f"=== seed model={model_seed} train={train_rng_seed} ===")
        steps, coincidental, _ = run_seed(model_seed, train_rng_seed)
        final = coincidental[-1]
        t_step = _transition_step(steps, coincidental, chance)
        final_coincidental.append(final)
        transition_steps.append(t_step)
        note = "" if t_step is not None else "  (final value does not clear chance -- no meaningful transition step)"
        print(f"  final coincidental accuracy: {final:.4f}")
        print(f"  transition step (halfway chance->final): {t_step}{note}\n")

    final_coincidental = np.array(final_coincidental)
    mean_final = final_coincidental.mean()
    std_final = final_coincidental.std()
    n_above_chance = int((final_coincidental > chance).sum())
    n_above_1layer = int((final_coincidental > BASELINE_1LAYER_COINCIDENTAL).sum())

    print("=== summary across seeds ===")
    print(f"final coincidental accuracy: mean={mean_final:.4f} std={std_final:.4f} "
          f"n_seeds={len(SEEDS)}")
    print(f"seeds above chance ({chance:.4f}): {n_above_chance}/{len(SEEDS)}")
    print(f"seeds above trained 1-layer baseline ({BASELINE_1LAYER_COINCIDENTAL:.4f}): "
          f"{n_above_1layer}/{len(SEEDS)}")
    print(f"transition steps: {transition_steps}")

    above_chance_verdict = "reliable" if n_above_chance == len(SEEDS) else "not reliable"
    above_baseline_verdict = "reliable" if n_above_1layer == len(SEEDS) else "not reliable"
    print(f"\ncore_claim: coincidental induction above chance is {above_chance_verdict} "
          f"across seeds ({n_above_chance}/{len(SEEDS)})")
    print(f"core_claim: coincidental induction above the trained 1-layer baseline is "
          f"{above_baseline_verdict} across seeds ({n_above_1layer}/{len(SEEDS)})")

    print("\n=== leakage check ===")
    collisions, total_rows = check_no_leakage([s[1] for s in SEEDS], NUM_TRAIN_STEPS)
    print(f"checked {total_rows} training-batch rows across {len(SEEDS)} seeds against "
          f"the {EVAL_SEED=} held-out batch")
    print(f"leakage_check: collisions={collisions} -> "
          f"{'CLEAN' if collisions == 0 else 'LEAK DETECTED'}")


if __name__ == "__main__":
    main()
