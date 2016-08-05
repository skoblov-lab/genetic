from collections.abc import Sequence, Callable
from genetic import recombination as rec
import abc

import numpy as np


__all__ = ["BaseIndividual", "SingleChromosomeIndividual"]


class BaseIndividual(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __eq__(self, other):
        """
        :type other: SingleChromosomeIndividual
        :rtype: bool
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __ne__(self, other):
        """
        :type other: SingleChromosomeIndividual
        :rtype: bool
        """
        raise NotImplementedError

    @abc.abstractmethod
    def replicate(self, *args, **kwargs):
        """
        :return: Should return a copy of one's genome
                 (with or without new mutations)
        :rtype: Sequence[Any]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def mate(self, other, *args, **kwargs):
        """
        :return: the result of mating with another BaseIndividual instance
        of the same type
        :rtype: BaseIndividual
        """
        raise NotImplementedError

    @abc.abstractproperty
    def engine(self):
        """
        :return: one's evolution engine
        :rtype: Union[Sequence[Callable], Sequence[Sequence[Callable]]]
        """
        raise NotImplementedError

    @abc.abstractproperty
    def genome(self):
        """
        :return: one's features
        :rtype: Sequence[Any]
        """
        raise NotImplementedError

    @abc.abstractproperty
    def mutrate(self):
        """
        :return: one's random mutation rate
        :rtype: float
        """
        raise NotImplementedError


class SingleChromosomeIndividual(BaseIndividual):
    """
    This is a basic single chromosome-individual

    :type _l: int
    :type _engine: Sequence[Callable]
    :type _genome: tuple
    """

    __slots__ = ("_l", "_engine", "_mutrate")

    def __init__(self, engine, mutrate, l=None, starting=None):
        # TODO add docs about starting_chromosome
        """
        :type engine: Union[Sequence[Callable[Optional]], Callable[Optional]]
        :param engine: an engine is used to mutate individual's features.
                       It can be:
                       - A single callable object that takes one argument
                         (current value of a feature or the entire chromosome)
                         and returns an new value.
                         It must handle `None` inputs if `starting_chromosome`
                         is `None`.
                       - A sequence of such callable object, one per each
                         feature
                       Note: when `genome` is `None` the first time
                       `engine` is used its callables get `None` as input,
                       which means they should return some starting value; this
                       detail may not be important for your implementation, (e.g.
                       if you don't have any special rules for starting values
                       and mutated values), but make sure the your callables
                       handle `None` inputs if you don't `starting_chr`
        :type mutrate: float
        :param mutrate: the binomial random mutation rate for each gene
        :type l: Optional[int]
        :param l: the number of features, defaults to `None`. Can't be == 0.
                  - If `l` is set to `None`, it's true value is inferred from
                    `len(engine)`, hence a `ValueError` will be raised if
                    `engine` is not a sequence;
                  - If `l` is not `None` and `engine` is a sequence such that
                    `len(engines) != l`, a `ValueError` will be raised.
        :type starting: Sequence[Any]
        :param starting: the genome to begin with
        :raise ValueError: If `l` is set to `None` and `engine` is not a
                           sequence;
        """

        if not isinstance(mutrate, float) or not (0 <= mutrate <= 1):
            raise ValueError("`mutrate` must be a float in [0, 1]")

        if l is not None and not isinstance(l, int) and l <= 0:
            raise ValueError("`l` is an optional positive integer")

        if l and isinstance(engine, Sequence) and len(engine) != l:
            raise ValueError("len(engine) != l, while l is not None")

        if l is None and not isinstance(engine, Sequence):
            raise ValueError("`l` is not specified, while `engine` is not a "
                             "sequence, hence `l` cannot be be inferred")

        if not isinstance(engine, Callable) and not (
                isinstance(engine, Sequence) and
                all(isinstance(gen, Callable) for gen in engine)):
            raise ValueError("`engine` must be a Callable or a Sequence of "
                             "Callable objects")

        self._engine = (engine if isinstance(engine, Sequence) else
                        [engine] * l)

        self._l = l if l else len(engine)

        if starting and (not isinstance(starting, Sequence) or
                         self._l != len(starting)):
            raise ValueError("`starting` length doesn't match the number "
                             "of features specified by `l` (or inferred from "
                             "`len(engine)`)")

        # chromosome is a sequence of genes (features)
        self._genome = (tuple(starting) if starting else
                        tuple(gen(None) for gen in self._engine))

        self._mutrate = mutrate

    def __eq__(self, other):
        """
        :type other: SingleChromosomeIndividual
        """
        return self.genome == other.genome

    def __ne__(self, other):
        """
        :type other: SingleChromosomeIndividual
        """
        return not self == other

    @property
    def engine(self):
        """
        :rtype: Sequence[Callable]
        """
        return self._engine

    @property
    def genome(self):
        """
        :rtype: tuple[Any]
        """
        return self._genome

    @property
    def mutrate(self):
        return self._mutrate

    def replicate(self):
        """
        :rtype: list[Any]
        :return: a mutated chromosome
        """
        mutation_mask = np.random.binomial(1, self.mutrate, len(self.genome))
        return [gen(val) if mutate else val for (val, mutate, gen) in
                list(zip(self.genome, mutation_mask, self.engine))]

    def mate(self, other, *args, **kwargs):
        """
        :type other: SingleChromosomeIndividual
        """
        offspring_genome = rec.binomial(self.replicate(), other.replicate())

        return type(self)(engine=self._engine, mutrate=self.mutrate,
                          l=self._l, starting=offspring_genome)


if __name__ == "__main__":
    raise RuntimeError
