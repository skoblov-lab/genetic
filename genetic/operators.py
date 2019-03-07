# TODO add exceptions to documentation

import operator as op
import random
from collections import Counter
from itertools import chain, starmap
from typing import Callable, Collection, Dict, Generic, List, Optional, Tuple, \
    TypeVar

import numpy as np  # type: ignore
from tqdm import tqdm  # type: ignore

from genetic.base import ChildMutator, Crossover, Estimator, Executor, \
    KeyType, MateSelector, Ord, Recorder, SelectionPolicy
from genetic.utils import replace


Individual = TypeVar('Individual', covariant=True)
Record = TypeVar('Record', covariant=True)

__all__ = (
    'GenericEstimator',
    'GenericRecorder',
    'GenericSelector',
    'GenericCrossover',
    'GenericChildMutator',
    'GenericPolicy',
    'ktournament'
)

# TODO should we reimplement Generic* in terms of the Functor API? Consider that
#      Python does not support higher-kinded types


# TODO abstract away cache access to make implementing different caches easy
class GenericEstimator(Generic[Individual], Estimator):

    def __init__(self,
                 func: Callable[[Individual], Ord],
                 cache=False,
                 executor: Optional[Executor] = None):
        if not callable(func):
            raise ValueError
        if not (executor is None or isinstance(executor, Executor)):
            raise ValueError
        self._func = func
        self.executor = executor
        self._cache: Optional[Dict[Individual, Ord]] = {} if cache else None

    @property
    def func(self) -> Callable[[Individual], Ord]:
        return self.func

    @property
    def cache(self) -> Optional[Dict[Individual, Ord]]:
        return self._cache

    def __call__(self, individuals: List[Individual], verbose=False, **kwargs) -> List[Ord]:
        """
        :param individuals:
        :param verbose:
        :param kwargs: these are present for API compatibility: they are not
        passed to the fitness function
        :return:
        >>> from copy import deepcopy
        >>> import numpy as np
        >>> identity = lambda x: x
        >>> estimator = GenericEstimator(identity, False)
        >>> individuals = list(range(10000000))
        >>> scores = estimator(individuals)
        >>> estimator.cache is None
        True
        >>> len(scores) == len(individuals)
        True
        >>> estimator = GenericEstimator(identity, True)
        >>> scores = estimator(individuals)
        >>> estimator.cache is None
        False
        >>> len(estimator.cache) == len(individuals)
        True
        >>> individuals = list(np.random.randint(0, 100000000, 1000))
        >>> estimator = GenericEstimator(identity, True)
        >>> scores = estimator(individuals)
        >>> len(estimator.cache) == len(set(individuals))
        True
        >>> old_cache = deepcopy(estimator.cache)
        >>> scores = estimator(individuals)
        >>> estimator(individuals, verbose=True) == individuals
        True
        >>> len(scores) == len(individuals)
        True
        >>> all(score == indiv for score, indiv in zip(scores, individuals))
        True
        >>> new_cache = deepcopy(estimator.cache)
        >>> old_cache == new_cache
        True
        >>> new_individuals = list(np.random.randint(0, 200000000, 1000))
        >>> scores = estimator(new_individuals)
        >>> new_cache = deepcopy(estimator.cache)
        >>> old_cache == new_cache
        False
        >>> (set(new_cache) - set(old_cache)) == (set(new_individuals) - set(individuals))
        True
        >>> len(scores) == len(new_individuals)
        True
        """
        map_ = map if self.executor is None else self.executor.map
        # lookup values in the cache if it's enabled and nonempty
        n_total = len(individuals)
        try:
            cached_scores: List[Optional[Ord]] = (
                [None] * n_total if not self._cache else
                list(map(self._cache.get, individuals))
            )
        except TypeError:
            raise TypeError('Caching requires hashable individuals')
        uncached = (
            individuals if not self._cache else
            [indiv for indiv, score in zip(individuals, cached_scores) if score is None]
        )
        n_cached = n_total - len(uncached)
        # calculate missing scores
        if verbose:
            msg = 'Estimating fitness'
            with tqdm(uncached, initial=n_cached, total=n_total, desc=msg) as uncached_:
                uncached_scores = list(map_(self._func, uncached_))  # type: ignore
        else:
            uncached_scores = list(map_(self._func, uncached))  # type: ignore
        # update cache
        if self._cache is not None and uncached:
            self._cache.update(
                {indiv: score for indiv, score in zip(uncached, uncached_scores)}
            )
        mask = [score is None for score in cached_scores] if uncached else None
        return (
            # all scores were cached
            cached_scores if not uncached_scores else
            # some scores were cached
            replace(mask, cached_scores, uncached_scores) if n_cached else
            # all scores were new
            uncached_scores
        )


class GenericRecorder(Generic[Individual, Record], Recorder):
    """
    Stores fitness, age and the number of children
    >>> import numpy as np
    >>> from itertools import chain
    >>> from collections import namedtuple, Counter
    >>> record = namedtuple('record', ['score', 'age', 'nchild'])
    >>> start = lambda indiv, score: record(score, 0, 0)
    >>> update = lambda indiv, rec, nchild: record(rec.score, rec.age+1, rec.nchild+nchild)
    >>> individuals = list(np.random.randint(-1000, 1000, size=100))
    >>> scores = individuals
    >>> recorder = GenericRecorder(start, update)
    >>> # test record initialisation
    >>> records = recorder.start(individuals, scores)
    >>> all(rec.score == score for rec, score in zip(records, scores))
    True
    >>> set((rec.age, rec.nchild) for rec in records) == {(0, 0)}
    True
    >>> # test record update
    >>> mates = [[0, 1], [0, 2], [1, 3]]
    >>> mate_counts = Counter(chain.from_iterable(mates))
    >>> broodsize = 1
    >>> updated1 = recorder.update(individuals, records, mates, broodsize)
    >>> len(updated1) == len(records)
    True
    >>> all(old.score == new.score for old, new in zip(records, updated1))
    True
    >>> all(new.age == 1 for new in updated1)
    True
    >>> all(updated1[key].nchild == (count * broodsize) for key, count in mate_counts.items())
    True
    >>> broodsize = 3
    >>> updated2 = recorder.update(individuals, records, mates, broodsize)
    >>> all(updated2[key].nchild == (count * broodsize) for key, count in mate_counts.items())
    True
    >>> all((new1.score, new1.age) == (new2.score, new2.age) for new1, new2 in zip(updated1, updated2))
    True
    """

    def __init__(self,
                 start: Callable[[Individual, Ord], Record],
                 update: Callable[[Individual, Record, int], Record]):
        """

        :param start: a function of two arguments: individual and fitness score
        :param update: a function of three arguments: individual, record and
        the number of new children
        """
        if not all(map(callable, [start, update])):
            raise ValueError
        self._start = start
        self._update = update

    def start(self,
              individuals: List[Individual],
              scores: List[Ord],
              **kwargs) -> List[Record]:
        """
        :param individuals:
        :param scores:
        :param kwargs:
        :return:
        """

        if not (individuals and len(individuals) == len(scores)):
            raise ValueError
        return list(starmap(self._start, zip(individuals, scores)))

    def update(self,
               individuals: List[Individual],
               records: List[Record],
               mates: List[List[int]],
               broodsize: int,
               **kwargs) -> List[Record]:

        if not (individuals and len(individuals) == len(records)):
            raise ValueError
        if not (isinstance(broodsize, int) and broodsize > 0):
            raise ValueError('broodsize must be a positive integer')
        # multiply (i.e. repeat) mating groups by broodsize and count the number
        # of children
        nchildren = Counter(
            chain.from_iterable(
                group * broodsize for group in mates
            )
        )
        # keys = range(len(individuals))  # TODO: should Recorder be verbose?
        return [
            self._update(indiv, record, nchildren[i])
            for i, (indiv, record) in enumerate(zip(individuals, records))
        ]


class GenericSelector(Generic[Individual], MateSelector):

    def __init__(self,
                 selector: Callable[[List[Individual], List[Record]], List[int]],
                 nmates: int):
        if not callable(selector):
            raise ValueError
        if not (isinstance(nmates, int) and nmates > 0):
            raise ValueError('nmates must be a positive integer')
        self._selector = selector
        self._nmates = nmates

    @property
    def nmates(self) -> int:
        return self._nmates

    def __call__(self,
                 npairs: int,
                 individuals: List[Individual],
                 records: List[Record],
                 verbose: bool = False,
                 **kwargs) -> List[List[int]]:
        """

        :param npairs:
        :param individuals:
        :param records:
        :param verbose:
        :param kwargs:
        :return:
        >>> import operator as op
        >>> import numpy as np
        >>> from fn import F
        >>> npairs = 10
        >>> nmates = 2
        >>> k = 10
        >>> key = op.itemgetter(0)
        >>> reverse = False
        >>> replace = False
        >>> func = F(ktournament, k, key, nmates, reverse=reverse, replace=replace)
        >>> individuals = list(np.random.randint(-1000, 1000, size=100))
        >>> records = [[indiv] for indiv in individuals]
        >>> selector = GenericSelector(func, nmates)
        >>> pairs = selector(npairs, individuals, records)
        >>> len(pairs) == npairs
        True
        >>> set(map(len, pairs)) == {2}
        True
        >>> pairs = selector(npairs, individuals, records, verbose=True)
        >>> selector = GenericSelector(func, 3)
        >>> selector(npairs, individuals, records, verbose=True)
        Traceback (most recent call last):
        RuntimeError: expected to select 3 individuals per mating group, but produced groups of [2] individuals
        """
        if not (isinstance(npairs, int) and npairs > 0):
            raise ValueError('npairs must be a positive integer')
        if not (individuals and len(individuals) == len(records)):
            raise ValueError(
                'either no individuals were passed or the number of individuals '
                'does not match the number of records'
            )
        counter = range(npairs)
        if verbose:
            msg = 'Selecting mates'
            with tqdm(counter, desc=msg) as counter_:
                mates = [self._selector(individuals, records) for _ in counter_]
        else:
            mates = [self._selector(individuals, records) for _ in counter]
        groupsizes = set(map(len, mates))
        if groupsizes != {self._nmates}:
            raise RuntimeError(
                f'expected to select {self._nmates} individuals per mating '
                f'group, but produced groups of {sorted(groupsizes)} individuals'
            )
        return mates


class GenericCrossover(Generic[Individual], Crossover):

    def __init__(self,
                 crossover: Callable[[List[Tuple[Individual, Record]]], List[Individual]],
                 nmates: int,
                 broodsize: int):
        """
        :param crossover: a callable that takes a list of `nmates` parents and
        returns a list of `broodsize` Individuals (children). Parents are given
        as tuples of form (Individual, Record).
        """
        if not callable(crossover):
            raise ValueError
        if not (isinstance(nmates, int) and nmates > 0):
            raise ValueError
        if not (isinstance(broodsize, int) and broodsize > 0):
            raise ValueError
        self._crossover = crossover
        self._nmates = nmates
        self._broodsize = broodsize

    @property
    def nmates(self) -> int:
        return self._nmates

    @property
    def broodsize(self) -> int:
        return self._broodsize

    def __call__(self,
                 individuals: List[Individual],
                 records: List[Record],
                 mates: Collection[Collection[KeyType]],
                 **kwargs) -> List[Individual]:
        """
        :param individuals:
        :param records:
        :param mates:
        :param kwargs:
        :return:
        >>> from itertools import product
        >>> import numpy as np
        >>>
        >>> def func(parents: List[Tuple[Tuple[int, int], None]]) -> List[Tuple[int, int]]:
        ...     (indiv1, rec1), (indiv2, rec2) = parents
        ...     assert {rec1, rec2} == {None}
        ...     child = (indiv1[0], indiv2[1])
        ...     return [child]
        ...
        >>> nmates = 2
        >>> broodsize = 1
        >>> individuals = [(0, 0), (1, 1)]
        >>> records = [None, None]
        >>> mates = list(product([0, 1], [0, 1]))
        >>> crossover = GenericCrossover(func, nmates, broodsize)
        >>> children = crossover(individuals, records, mates)
        >>> set(children) == set(mates)
        True
        >>> def func(parents: List[Tuple[Tuple[int, int], None]]) -> List[Tuple[int, int]]:
        ...     (indiv1, rec1), (indiv2, rec2) = parents
        ...     assert {rec1, rec2} == {None}
        ...     return list(product(indiv1, indiv2))
        ...
        >>> crossover = GenericCrossover(func, nmates, broodsize)
        >>> crossover(individuals, records, mates)
        Traceback (most recent call last):
        RuntimeError: expected to produce 1 children per mating pair; the underlying function produced [4] children instead
        """
        if not (individuals and len(individuals) == len(records)):
            raise ValueError
        if not (mates and set(map(len, mates)) == {self.nmates}):
            raise ValueError(
                f'mates must be a nonempty sequence of sequences, {self.nmates} '
                f'indices each'
            )
        parents = (
            [(individuals[key], records[key]) for key in group]
            for group in mates
        )
        child_groups = list(map(self._crossover, parents))
        broodsizes = set(map(len, child_groups))
        if broodsizes != {self.broodsize}:
            raise RuntimeError(
                f'expected to produce {self.broodsize} children per mating '
                f'pair; the underlying function produced {list(broodsizes)} '
                f'children instead'
            )
        return list(chain.from_iterable(child_groups))


class GenericChildMutator(Generic[Individual], ChildMutator):

    def __init__(self, mutator: Callable[[Individual], Individual]):
        if not callable(mutator):
            raise ValueError
        self._mutator = mutator

    def __call__(self, individuals: List[Individual], **kwargs) -> List[Individual]:
        if not individuals:
            raise ValueError
        return list(map(self._mutator, individuals))


class GenericPolicy(Generic[Individual, Record], SelectionPolicy):

    def __init__(self, selector: Callable[[int, List[Individual], List[Record]], List[int]]):
        """
        :param selector: a function of three arguments: how many individuals to
        select, a list of individuals, a list of records; the function must return
        a list of selected indices
        """
        if not callable(selector):
            raise ValueError
        self._selector = selector

    def __call__(self,
                 size: int,
                 individuals: List[Individual],
                 records: List[Record],
                 **kwargs) -> List[int]:
        if not (isinstance(size, int) and size > 0):
            raise ValueError('size must be a positive integer')
        if not (individuals and len(individuals) == len(records)):
            raise ValueError
        # note: we do not require selected indices to be unique, hence we do not
        # check, whether len(individuals) >= size
        selected = self._selector(size, individuals, records)
        if len(selected) != size:
            raise RuntimeError(
                f'expected to select {size} individuals, but selected '
                f'{len(selected)} individuals instead'
            )
        return selected


def ktournament(k: int, key: Callable[[Record], Ord], n: int, individuals: List[Individual], records: List[Record],
                reverse=False, replace=False) -> List[int]:
    """
    K-tournament selection
    :param n: the number of tournaments to run
    :param k: how many participants are randomly selected for the tournament
    :param key: a mapping between records and an Orderable type
    :param individuals:
    :param records:
    :param reverse: by default the function selects an individual with the
    highest key(record) values; passing True reverses selection from min to max
    :param replace: return selected indices into the selection pool, i.e. allow
    duplicate indices in the output
    :return: index of a selected individual
    >>> import operator as op
    >>> import numpy as np
    >>> n = 100
    >>> k = 2
    >>> key = op.itemgetter(0)
    >>> reverse = False
    >>> replace = False
    >>> individuals = list(np.random.randint(-1000, 1000, size=100))
    >>> records = [[indiv] for indiv in individuals]
    >>> ktournament(k,key,n,individuals,records,reverse,replace)
    Traceback (most recent call last):
    ValueError: insufficient initial population size for given arguments n, k and replace
    >>> n = 99
    >>> selected = ktournament(k,key,n,individuals,records,reverse,replace)
    >>> len(set(selected)) == n
    True
    >>> k = 100
    >>> n = 1
    >>> nrep = 100
    >>> selection_max = np.array([
    ...     individuals[ktournament(k,key,n,individuals,records,reverse,replace)[0]]
    ...     for _ in range(nrep)
    ... ])
    >>> (selection_max == max(individuals)).all()
    True
    >>> reverse = True
    >>> selection_min = np.array([
    ...     individuals[ktournament(k,key,n,individuals,records,reverse,replace)[0]]
    ...     for _ in range(nrep)
    ... ])
    >>> (selection_min == min(individuals)).all()
    True
    >>> k = 2
    >>> n = 100
    >>> replace = True
    >>> selected = ktournament(k,key,n,individuals,records,reverse,replace)
    >>> len(set(selected)) < n
    True
    >>> calculate_mean = lambda indices: np.array(individuals)[indices].mean()
    >>> k = 100
    >>> n = 1000
    >>> nrep = 100
    >>> reverse = False
    >>> selection_means = np.array([
    ...     calculate_mean(ktournament(k,key,n,individuals,records,reverse,replace))
    ...     for _ in range(nrep)
    ... ])
    >>> (selection_means > np.mean(individuals)).sum() / nrep > 0.9
    True
    >>> reverse = True
    >>> selection_means = np.array([
    ...     calculate_mean(ktournament(k,key,n,individuals,records,reverse,replace))
    ...     for _ in range(nrep)
    ... ])
    >>> (selection_means < np.mean(individuals)).sum() / nrep > 0.9
    True
    """
    if not (isinstance(n, int) and n > 0):
        raise ValueError
    if not (len(individuals) and len(individuals) == len(records)):
        raise ValueError
    sample_size_limit = len(individuals) if replace else len(individuals) - n + 1
    if not (isinstance(k, int) and k <= sample_size_limit):
        raise ValueError(
            'insufficient initial population size for given arguments n, k and '
            'replace'
        )
    selector = min if reverse else max
    key_ = lambda i: key(records[i])
    if replace:
        # this is somewhat efficient
        indices = range(len(individuals))
        samples = (random.sample(indices, k) for _ in range(n))
        selected = [selector(sample, key=key_) for sample in samples]
    else:
        # this is quite inefficient; I've found no better alternatives based on
        # built-in and/or numpy functions; we might write our own implementation
        # in Cython
        indices = np.arange(len(individuals))
        available = np.ones_like(indices, dtype=bool)
        selected = []
        for _ in range(n):
            sample = np.random.choice(indices[available], k, replace=False)
            selected_idx = selector(sample, key=key_)
            selected.append(selected_idx)
            available[selected_idx] = False
    return selected


if __name__ == '__main__':
    raise RuntimeError
