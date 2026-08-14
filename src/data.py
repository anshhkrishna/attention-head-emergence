"""Synthetic in-context copying task: a random prefix followed by an exact repeat of
itself, plus an induction mask marking the positions where the repeat lets next-token
prediction beat chance.

A purely i.i.d. sequence carries no such signal -- if a token recurs by chance, what
followed it the first time is independent of what follows it the second time, since
every draw is independent. The prefix-then-repeat construction (the standard synthetic
task in the induction-head literature this project is testing, e.g. Elhage et al. 2021
and Olsson et al. 2022) fixes that: the second copy is a literal repeat of the first, so
the token that follows a given value is *guaranteed* to be the same both times it
occurs at corresponding positions. Everything is still drawn from NumPy's RNG in-process
-- no download, no external service, no cost.
"""
import numpy as np

VOCAB_SIZE = 12
HALF_LEN = 16
SEQ_LEN = 2 * HALF_LEN
REFERENCE_SEED = 0
EVAL_SEED = 1


def make_batch(batch_size, seed, vocab_size=VOCAB_SIZE, half_len=HALF_LEN):
    """Return (tokens, mask): tokens is (batch_size, 2 * half_len) int array, each row
    a random prefix of length half_len concatenated with an exact copy of itself; mask
    is the induction_mask of each row.
    """
    rng = np.random.default_rng(seed)
    prefix = rng.integers(0, vocab_size, size=(batch_size, half_len))
    tokens = np.concatenate([prefix, prefix], axis=1)
    mask = np.stack([induction_mask(row) for row in tokens])
    return tokens, mask


def induction_mask(seq):
    """For each position t, True if seq[t] occurred at some earlier position s < t.

    At such a position, the "induction" prediction target is seq[s+1] for the most
    recent such s (attend to the last occurrence and copy what followed it) -- for the
    prefix-then-repeat construction, positions in the second half always have such an
    earlier occurrence, and the canonical one (the corresponding position in the first
    copy) is guaranteed to be followed by the same token both times.
    """
    n = len(seq)
    mask = np.zeros(n, dtype=bool)
    seen = set()
    for t in range(n):
        if seq[t] in seen:
            mask[t] = True
        seen.add(seq[t])
    return mask


if __name__ == "__main__":
    tokens, mask = make_batch(4, seed=REFERENCE_SEED)
    print(f"tokens: shape={tokens.shape} vocab=[0, {VOCAB_SIZE})")
    print(f"induction positions per sequence: {mask.sum(axis=1).tolist()} / {SEQ_LEN}")
