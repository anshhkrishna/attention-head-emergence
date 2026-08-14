import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import EVAL_SEED, HALF_LEN, REFERENCE_SEED, SEQ_LEN, VOCAB_SIZE, induction_mask, make_batch


def test_shape_and_range():
    tokens, mask = make_batch(64, seed=REFERENCE_SEED)
    assert tokens.shape == (64, SEQ_LEN)
    assert mask.shape == (64, SEQ_LEN)
    assert tokens.min() >= 0 and tokens.max() < VOCAB_SIZE
    assert mask.dtype == bool


def test_determinism():
    a, mask_a = make_batch(32, seed=7)
    b, mask_b = make_batch(32, seed=7)
    assert np.array_equal(a, b)
    assert np.array_equal(mask_a, mask_b)


def test_different_seeds_differ():
    a, _ = make_batch(32, seed=REFERENCE_SEED)
    b, _ = make_batch(32, seed=EVAL_SEED)
    assert not np.array_equal(a, b)


def test_second_half_is_a_repeat():
    tokens, _ = make_batch(16, seed=REFERENCE_SEED)
    assert np.array_equal(tokens[:, :HALF_LEN], tokens[:, HALF_LEN:])


def test_second_half_always_marked_induction():
    # every position in the repeated half has occurred earlier, by construction.
    tokens, mask = make_batch(16, seed=REFERENCE_SEED)
    assert mask[:, HALF_LEN:].all()


def test_duplicate_boundary_continuation_matches():
    # the token following a given position in the repeat equals the token following
    # the corresponding position in the original prefix -- the guaranteed signal an
    # induction head exploits.
    tokens, _ = make_batch(16, seed=REFERENCE_SEED)
    first_copy_next = tokens[:, 1:HALF_LEN]
    second_copy_next = tokens[:, HALF_LEN + 1:]
    assert np.array_equal(first_copy_next, second_copy_next)


def test_induction_mask_hand_worked_examples():
    # seq: 3 5 3 5 5 3
    # t=0 (3): first occurrence -> False
    # t=1 (5): first occurrence -> False
    # t=2 (3): seen at t=0 -> True, target = seq[0 + 1] = 5
    # t=3 (5): seen at t=1 -> True, target = seq[1 + 1] = 3
    # t=4 (5): most recent occurrence of 5 is t=3 -> True, target = seq[3 + 1] = 5
    # t=5 (3): most recent occurrence of 3 is t=2 -> True, target = seq[2 + 1] = 5
    seq = np.array([3, 5, 3, 5, 5, 3])
    expected = np.array([False, False, True, True, True, True])
    assert np.array_equal(induction_mask(seq), expected)


def test_induction_mask_no_repeats():
    seq = np.array([0, 1, 2, 3, 4])
    assert not induction_mask(seq).any()


def test_induction_mask_all_same_token():
    seq = np.array([9, 9, 9, 9])
    expected = np.array([False, True, True, True])
    assert np.array_equal(induction_mask(seq), expected)
