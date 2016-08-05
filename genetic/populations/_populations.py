from collections.abc import Sequence, Callable
from operator import itemgetter
from random import sample
import itertools
import abc

import numpy as np

from genetic.individuals import BaseIndividual
from genetic.util import Workers, filter_duplicates

__all__ = ["BasePopulation", "PanmicticPopulation"]


class BasePopulation:

    @abc.abstractmethod
    def evolve(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractproperty
    def individuals(self):
        """
        :rtype: Sequence[(Any, BaseIndividual)]
        """
        raise NotImplementedError

    @abc.abstractproperty
    def legends(self):
        """
        :rtype: Sequence[(Any, BaseIndividual)]
        """
        raise NotImplementedError

    @abc.abstractproperty
    def nlegends(self):
        """
        :rtype: int
        """
        raise NotImplementedError

    @abc.abstractproperty
    def fitness_func(self):
        """
        :rtype: Callable
        """
        raise NotImplementedError

    @abc.abstractproperty
    def selection(self):
        """
        :rtype: Callable
        """
        raise NotImplementedError

    @abc.abstractproperty
    def size(self):
        raise NotImplementedError


class PanmicticPopulation(BasePopulation):
    """
    :type _individuals: list[(Any, BaseIndividual)]
    :type _legends: list[(Any, BaseIndividual)]
    :type _fitness_func: (BaseIndividual) -> Any
    """

    def __init__(self, ancestors, size, fitness, selection, nlegends=100):
        """
        :type ancestors: Sequence[BaseIndividual]
        :param ancestors: a bunch of individuals to begin with
        :type size: int
        :param size: population size
        :type fitness: (BaseIndividual) -> Any
        :param fitness: a callable that requires one argument - an
                        instance of Individual - and returns an
                        instance of a class that supports comparison
                        operators, i.e. can be used to evaluate and
                        compare fitness of different Individuals.
        :type selection: Callable
        :param selection: a selection engine
        :type nlegends: int
        :param nlegends: the number of legends to remember
        :return:
        """
        if not isinstance(nlegends, int) or nlegends < 0:
            raise ValueError("`n_legends` must be a non-negative integer")

        if not isinstance(size, int) or size <= 0:
            raise ValueError("`size` must be a positive integer")

        if not isinstance(ancestors, Sequence) or len(ancestors) < 2:
            raise ValueError("`ancestors` must be a nonempty sequence of "
                             "length >= 2")

        if not all(isinstance(indiv, BaseIndividual) for indiv in ancestors):
            raise ValueError("`ancestors` can only contain instances of"
                             "`Individual`")
        try:
            if fitness(ancestors[0]) is None:
                raise ValueError("`fitness_function` mustn't return `NoneType` "
                                 "values")
        except (TypeError, AttributeError):
            raise ValueError("Your `fitness` doesn't suit your Idividuals")

        self._size = size
        self._fitness_func = fitness
        self._selection = selection
        self._nlegends = nlegends
        self._evolving = False
        self._individuals = list(zip(map(fitness, ancestors), ancestors))
        self._legends = []

    @property
    def size(self):
        return self._size

    @property
    def legends(self):
        """
        :rtype: list[(Any, BaseIndividual)]
        """
        return self._legends

    @property
    def nlegends(self):
        return self._nlegends

    @property
    def individuals(self):
        return self._individuals

    @property
    def fitness_func(self):
        return self._fitness_func

    @property
    def selection(self):
        return self._selection

    def evolve(self, n, jobs=1):
        """
        :rtype: Generator[Any]
        """

        if not isinstance(jobs, int) or jobs < 1:
            raise ValueError("`jobs` must be a positive integer")

        def repopulate(evaluated_individuals):
            """
            :type evaluated_individuals: list[(Any, BaseIndividual)]
            :rtype: list[(Any, BaseIndividual)]
            """
            def mate(indiv1, indiv2):
                """
                :type indiv1: (Any, BaseIndividual)
                :type indiv2: (Any, BaseIndividual)
                """
                return indiv1[1].mate(indiv2[1])

            n_pairs = self.size - len(evaluated_individuals)
            pairs = [sample(self.individuals, 2) for _ in range(n_pairs)]
            new_individuals = workers.starmap(mate, pairs)
            scores = workers.map(self.fitness_func, new_individuals)
            return evaluated_individuals + list(zip(scores, new_individuals))

        def new_legends(old_legends, contenders):
            """
            :type old_legends: list[(Any, BaseIndividual)]
            :type contenders: list[(Any, BaseIndividual)]
            """
            merged = sorted(filter_duplicates(old_legends + contenders),
                            key=itemgetter(0), reverse=True)
            return merged[:self.nlegends]

        workers = Workers(jobs)

        if len(self.individuals) < self.size:
            self._individuals = repopulate(self.individuals)

        for _ in itertools.repeat(None, n):
            survivors = self.selection(sorted(self.individuals,
                                              key=itemgetter(0), reverse=True))
            self._individuals = repopulate(survivors)
            # Update legends
            self._legends = new_legends(self.legends, self.individuals)
            yield np.mean([indiv[0] for indiv in self.individuals])

        workers.terminate()


if __name__ == "__main__":
    raise RuntimeError
