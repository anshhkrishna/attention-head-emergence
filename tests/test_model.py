"""Checks the hand-derived backward pass against numerical gradients from finite
differences, a couple of basic sanity properties of the forward pass, and the core
claim of the project as the multi-seed rigor sweep actually shows it.
"""
import re
from pathlib import Path

import numpy as np
import pytest

from model import backward, cross_entropy_loss, forward, init_params

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RIGOR_LOG = RESULTS_DIR / "rigor.log"

N_LAYERS = 2
D_MODEL = 4
SEQ_LEN = 6
VOCAB_SIZE = 5
BATCH_SIZE = 3
FD_EPS = 1e-5
SAMPLES_PER_ARRAY = 3
REL_ERROR_TOL = 1e-4


def _param_arrays(params, grads):
    """Flatten the nested params/grads structure into (param_array, grad_array,
    label) triples, so every weight matrix in the model can be checked the same
    way regardless of where it lives in the structure.
    """
    pairs = [
        (params["W_E"], grads["W_E"], "W_E"),
        (params["W_pos"], grads["W_pos"], "W_pos"),
        (params["W_U"], grads["W_U"], "W_U"),
    ]
    for i, (p_layer, g_layer) in enumerate(zip(params["layers"], grads["layers"])):
        for key in p_layer:
            pairs.append((p_layer[key], g_layer[key], f"layers[{i}].{key}"))
    return pairs


def test_backward_matches_finite_differences():
    params = init_params(N_LAYERS, D_MODEL, VOCAB_SIZE, SEQ_LEN, seed=42)
    rng = np.random.default_rng(7)
    tokens = rng.integers(0, VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))

    logits, cache = forward(params, tokens, cache=True)
    base_loss = cross_entropy_loss(logits, tokens)
    assert np.isfinite(base_loss)

    grads = backward(params, tokens, cache)

    def loss_of(p):
        return cross_entropy_loss(forward(p, tokens, cache=False), tokens)

    idx_rng = np.random.default_rng(3)
    max_rel_error = 0.0
    for arr, grad_arr, label in _param_arrays(params, grads):
        flat_indices = idx_rng.choice(arr.size, size=min(SAMPLES_PER_ARRAY, arr.size), replace=False)
        for flat in flat_indices:
            multi_index = np.unravel_index(flat, arr.shape)
            original = arr[multi_index]

            arr[multi_index] = original + FD_EPS
            loss_plus = loss_of(params)
            arr[multi_index] = original - FD_EPS
            loss_minus = loss_of(params)
            arr[multi_index] = original

            numeric_grad = (loss_plus - loss_minus) / (2 * FD_EPS)
            analytic_grad = grad_arr[multi_index]
            denom = max(abs(numeric_grad), abs(analytic_grad), 1e-8)
            rel_error = abs(numeric_grad - analytic_grad) / denom
            max_rel_error = max(max_rel_error, rel_error)

    assert max_rel_error < REL_ERROR_TOL, f"max relative error {max_rel_error} exceeds {REL_ERROR_TOL}"


def test_forward_output_shape_and_causality():
    params = init_params(N_LAYERS, D_MODEL, VOCAB_SIZE, SEQ_LEN, seed=42)
    rng = np.random.default_rng(1)
    tokens = rng.integers(0, VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))

    logits = forward(params, tokens, cache=False)
    assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    assert np.all(np.isfinite(logits))

    # Changing a later token must not change an earlier position's logits: the
    # causal mask should make position 0's prediction independent of position 5.
    tokens_perturbed = tokens.copy()
    tokens_perturbed[:, -1] = (tokens_perturbed[:, -1] + 1) % VOCAB_SIZE
    logits_perturbed = forward(params, tokens_perturbed, cache=False)
    np.testing.assert_allclose(logits[:, 0, :], logits_perturbed[:, 0, :])


SEED_HEADER_RE = re.compile(r"^=== seed model=(\d+) train=(\d+) ===$")
FINAL_COINCIDENTAL_RE = re.compile(r"^  final coincidental accuracy: (\d+\.\d+)$")
ABOVE_CHANCE_RE = re.compile(r"^seeds above chance \((\d+\.\d+)\): (\d+)/(\d+)$")
ABOVE_BASELINE_RE = re.compile(
    r"^seeds above trained 1-layer baseline \((\d+\.\d+)\): (\d+)/(\d+)$")
CORE_CLAIM_CHANCE_RE = re.compile(
    r"^core_claim: coincidental induction above chance is (reliable|not reliable) "
    r"across seeds \((\d+)/(\d+)\)$")
CORE_CLAIM_BASELINE_RE = re.compile(
    r"^core_claim: coincidental induction above the trained 1-layer baseline is "
    r"(reliable|not reliable) across seeds \((\d+)/(\d+)\)$")
LEAKAGE_RE = re.compile(r"^leakage_check: collisions=(\d+) -> (CLEAN|LEAK DETECTED)$")


def parse_rigor_log(path=RIGOR_LOG):
    """Parses results/rigor.log (produced by `python src/rigor.py`, a sweep training
    the 2-layer model across several seeds) rather than retraining inside the test
    suite -- the sweep takes over a minute, which would make pytest too slow to run
    routinely. This still checks real, committed numbers: if src/rigor.py is rerun and
    the log changes, this test checks the new log, not a value frozen at authoring
    time.
    """
    seeds = []
    final_coincidental = []
    above_chance = None
    above_baseline = None
    core_claim_chance = None
    core_claim_baseline = None
    leakage = None

    for line in path.read_text().splitlines():
        m = SEED_HEADER_RE.match(line)
        if m:
            seeds.append((int(m.group(1)), int(m.group(2))))
            continue
        m = FINAL_COINCIDENTAL_RE.match(line)
        if m:
            final_coincidental.append(float(m.group(1)))
            continue
        m = ABOVE_CHANCE_RE.match(line)
        if m:
            above_chance = (float(m.group(1)), int(m.group(2)), int(m.group(3)))
            continue
        m = ABOVE_BASELINE_RE.match(line)
        if m:
            above_baseline = (float(m.group(1)), int(m.group(2)), int(m.group(3)))
            continue
        m = CORE_CLAIM_CHANCE_RE.match(line)
        if m:
            core_claim_chance = (m.group(1), int(m.group(2)), int(m.group(3)))
            continue
        m = CORE_CLAIM_BASELINE_RE.match(line)
        if m:
            core_claim_baseline = (m.group(1), int(m.group(2)), int(m.group(3)))
            continue
        m = LEAKAGE_RE.match(line)
        if m:
            leakage = (int(m.group(1)), m.group(2))

    return {
        "seeds": seeds,
        "final_coincidental": final_coincidental,
        "above_chance": above_chance,
        "above_baseline": above_baseline,
        "core_claim_chance": core_claim_chance,
        "core_claim_baseline": core_claim_baseline,
        "leakage": leakage,
    }


def test_rigor_log_uses_at_least_three_seeds():
    log = parse_rigor_log()
    assert len(log["seeds"]) >= 3
    assert len(log["final_coincidental"]) == len(log["seeds"])


def test_no_training_batch_leaks_into_the_eval_batch():
    log = parse_rigor_log()
    collisions, verdict = log["leakage"]
    assert collisions == 0
    assert verdict == "CLEAN"


def test_core_claim_coincidental_induction_reliably_above_chance_across_seeds():
    """After training, coincidental-position induction accuracy (the metric that
    isolates genuine content-based composition from the positional shortcut a model
    can exploit at fixed-offset positions) clears chance in every seed. Checked two
    ways -- src/rigor.py's own logged verdict, and an independent recomputation from
    the parsed per-seed final accuracies, so a bug in how src/rigor.py prints its
    verdict can't silently pass this test.
    """
    log = parse_rigor_log()
    verdict, n_above, n_seeds = log["core_claim_chance"]
    assert n_seeds == len(log["seeds"])
    assert verdict == "reliable"
    assert n_above == n_seeds

    chance, logged_n_above, logged_n_seeds = log["above_chance"]
    assert (logged_n_above, logged_n_seeds) == (n_above, n_seeds)
    recomputed_n_above = sum(1 for acc in log["final_coincidental"] if acc > chance)
    assert recomputed_n_above == n_seeds


def test_core_claim_does_not_reliably_beat_the_1layer_baseline():
    """The 2-layer model's coincidental induction accuracy turns out statistically
    indistinguishable from the trained 1-layer baseline's (results/baseline.log)
    across seeds, not reliably higher -- checked honestly here rather than assumed.
    The 1-layer baseline itself does not perform genuine content-based induction (see
    results/baseline.log), so this is a negative result about the 2-layer model's
    composition mechanism, not a redefinition of what "beats the baseline" means.
    Checked two ways, as above.
    """
    log = parse_rigor_log()
    verdict, n_above, n_seeds = log["core_claim_baseline"]
    assert n_seeds == len(log["seeds"])
    assert verdict == "not reliable"
    assert n_above < n_seeds

    baseline_value, logged_n_above, logged_n_seeds = log["above_baseline"]
    assert (logged_n_above, logged_n_seeds) == (n_above, n_seeds)
    recomputed_n_above = sum(1 for acc in log["final_coincidental"] if acc > baseline_value)
    assert recomputed_n_above == n_above
    assert recomputed_n_above < n_seeds


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
