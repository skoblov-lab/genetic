from collections import Sequence
from itertools import starmap
import random

from multiprocess.pool import Pool

__all__ = ["Workers", "random_pairs", "filter_duplicates"]


class Workers(Pool):

    @property
    def processes(self):
        return self._processes

    def map(self, func, iterable, chunksize=None):
        if self.processes == 1:
            return list(map(func, iterable))
        return super().map(func, iterable, chunksize=chunksize)

    def starmap(self, func, iterable, chunksize=None):
        if self.processes == 1:
            return list(starmap(func, iterable))
        return super().starmap(func, iterable, chunksize=chunksize)


def random_pairs(sequence: Sequence, n: int):
    pairs = set()
    indices = list(range(len(sequence)))
    while len(pairs) != n:
        i, j = tuple(random.sample(indices, 2))
        if (i, j) in pairs or (j, i) in pairs:
            continue
        pairs.add((i, j))
    yield from ((sequence[i], sequence[j]) for i, j in pairs)


def filter_duplicates(objects):
    """
    Filter duplicated references from a sequence. Utilises object ids.
    :type objects: Sequence[Any]
    :rtype: list[Any]
    """
    return {id(obj): obj for obj in objects}.values()


if __name__ == "__main__":
    raise RuntimeError
