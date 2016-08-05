import numpy as np


__all__ = ["binomial"]


def binomial(chr1, chr2):
    """
    Picks one allele or the other with 50% success
    :type chr1: Sequence
    :type chr2: Sequence
    """
    if len(chr1) != len(chr2):
        raise ValueError("Incompatible chromosome lengths")
    choice_mask = np.random.binomial(1, 0.5, len(chr1))
    return [a if ch else b for (ch, a, b) in zip(choice_mask, chr1, chr2)]


def breakpoint(chr1, chr2, rate):
    raise NotImplemented


if __name__ == "__main__":
    raise RuntimeError
