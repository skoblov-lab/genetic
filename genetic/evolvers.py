from typing import List, Optional, Tuple
from itertools import chain

from genetic.base import BaseEvolver, Individual, \
    Operators, Record


# TODO add callbacks to the tests
# TODO kwargs in operators should be reserved for additional arguments
class GenericEvolver(BaseEvolver):
    """
    >>> import operator as op
    >>> from collections import namedtuple
    >>> from numbers import Number
    >>> import numpy as np
    >>> from fn import F
    >>> from genetic.operators import *
    >>> from genetic.base import Operators
    >>> # create a dataset
    >>> genindiv = lambda : np.random.randint(0, 2, size=1000).astype(bool)
    >>> individuals = [genindiv() for _ in range(1000)]
    >>> # create a recorder
    >>> Record = namedtuple('record', ['score', 'age'])
    >>> start_record = lambda indiv, score: Record(score, 0)
    >>> update_record = (
    ...     lambda indiv, rec: Record(rec[0], rec[1]+1)
    ... )
    >>> recorder = GenericRecorder(start_record, update_record)
    >>> # create an estimator
    >>> def fitness(indiv: np.ndarray) -> Number:
    ...     return indiv.sum()
    ...
    >>> estimator = GenericEstimator(fitness)
    >>> # create a mate selector
    >>> nmates = 2
    >>> selector = GenericSelector(
    ...     F(ktournament, 10, lambda x: x.score, nmates, reverse=True),
    ...     nmates
    ... )
    >>> # create a crossover operator
    >>> broodsize = 1
    >>> def discrete(parents: List[Tuple[np.ndarray, Record]]) -> List[np.ndarray]:
    ...     (i1, r1), (i2, r2) = parents
    ...     assert i1.shape == i2.shape
    ...     mask = np.random.binomial(True, 0.5, size=i1.shape).astype(bool)
    ...     return [np.where(mask, i1, i2)]
    ...
    >>> crossover = GenericCrossover(discrete, nmates, broodsize)
    >>> # create a mutation operator
    >>> def mutate(indiv: np.ndarray) -> np.ndarray:
    ...    mask = np.random.binomial(1, 0.0, indiv.shape).astype(bool)
    ...    return np.where(mask, ~indiv, indiv)
    ...
    >>> mutator = GenericMutator(mutate)
    >>> # create a selection policy
    >>> policy = GenericPolicy(
    ...     F(ktournament, 25, lambda x: x.score, reverse=True)
    ... )
    >>> # evolve a population from scratch for 10 generations
    >>> operators = Operators(
    ...     recorder, estimator, selector, crossover, mutator, policy
    ... )
    >>> evolver = GenericEvolver()
    >>> evolved1, records1 = evolver.evolve(
    ...     generations=1,
    ...     operators=operators,
    ...     gensize=200,
    ...     individuals=individuals,
    ...     verbose=False
    ... )
    >>> all(fitness(indiv) == rec.score for indiv, rec in zip(evolved1, records1))
    True
    >>> # continue the process multiple times to generate stats
    >>> evolved = [
    ...     evolver.evolve(generations=10, operators=operators, gensize=200,
    ...                    individuals=evolved1, records=records1, verbose=False)
    ...     for _ in range(50)
    ... ]
    >>> best = np.array([min(rec.score for rec in recs) for _, recs in evolved])
    >>> # this is not particularly deterministic
    >>> (best < min(rec.score for rec in records1)).sum() / len(best) >= 0.95
    True
    """

    def evolve_generation(
            self,
            operators: Operators,
            gensize: int,
            individuals: List[Individual],
            records: Optional[List[Record]] = None,
            **kwargs
    ) -> Tuple[List[Individual], List[Record]]:
        recorder, estimator, selector, crossover, mutator, policy = operators
        popsize = len(individuals)
        # compatibility check
        if not (isinstance(gensize, int) and gensize > 0):
            raise ValueError(
                'gensize must be a positive integer'
            )
        if selector.nmates != crossover.nmates:
            raise ValueError(
                'selector.nmates != crossover.nmates'
            )
        if gensize % crossover.broodsize:
            raise ValueError(
                'Generation size is not divisible by crossover.broodsize'
            )
        # start records if none were provided
        records = (
            records or
            recorder.start(individuals, estimator(individuals, **kwargs), **kwargs)
        )
        # create children and make sure everything goes as planned
        n_mating_groups = gensize // crossover.broodsize
        mates = selector(n_mating_groups, individuals, records, **kwargs)
        if len(mates) != n_mating_groups:
            raise RuntimeError
        recombined_groups = crossover(individuals, records, mates, **kwargs)
        recombined = list(chain.from_iterable(recombined_groups))
        if len(recombined) != n_mating_groups:
            raise RuntimeError
        children = mutator(recombined, **kwargs)
        child_scores = estimator(children, **kwargs)
        child_records = recorder.start(children, child_scores, **kwargs)
        if not (children and len(children) == len(child_records)):
            raise RuntimeError
        if len(children) != gensize:
            raise RuntimeError
        # update initial records
        records_updated = recorder.update(individuals, records)
        # join individuals and records and select a new population
        # todo move the join operation into a method
        joined_individuals = individuals + children
        joined_records = records_updated + child_records
        selected = policy(popsize, joined_individuals, joined_records)
        if len(selected) != popsize:
            raise RuntimeError
        return (
            [joined_individuals[i] for i in selected],
            [joined_records[i] for i in selected]
        )


if __name__ == '__main__':
    raise RuntimeError
