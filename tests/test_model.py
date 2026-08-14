"""Checks the hand-derived backward pass against numerical gradients from finite
differences, and a couple of basic sanity properties of the forward pass.
"""
import numpy as np
import pytest

from model import backward, cross_entropy_loss, forward, init_params

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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
