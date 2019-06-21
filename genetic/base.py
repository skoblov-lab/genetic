import abc
from itertools import chain
from functools import reduce
from typing import Callable, Collection, Generic, List, NamedTuple, \
    Optional, Tuple, TypeVar, Iterable, Sized, Container

KeyT = TypeVar('KeyT')
ValT = TypeVar('ValT', covariant=True)
Individual = TypeVar('Individual', covariant=True)
Record = TypeVar('Record', covariant=True)


class Ord(metaclass=abc.ABCMeta):
    """
    An abstract base class representing an orderable object, i.e. any object
    that satisfies the order axioms.
    """

    @abc.abstractmethod
    def __eq__(self, other):
        pass

    @abc.abstractmethod
    def __ne__(self, other):
        pass

    @abc.abstractmethod
    def __le__(self, other):
        pass

    @abc.abstractmethod
    def __lt__(self, other):
        pass

    @abc.abstractmethod
    def __ge__(self, other):
        pass

    @abc.abstractmethod
    def __gt__(self, other):
        pass

    @classmethod
    def __subclasshook__(cls, subclass):
        # TODO might be inefficient
        required = {'__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__'}
        # exclude methods inherited from `object`
        available = chain.from_iterable(
            base.__dict__ for base in subclass.__mro__ if base is not object
        )
        compatible = cls is Ord and not (required - set(available))
        return compatible or NotImplemented


class Index(Generic[KeyT, ValT], Sized, metaclass=abc.ABCMeta):
    """
    An abstract base class representing a finite set of objects of type ValT
    that can be accessed by an index/key of type KeyT. Examples include
    builtin arrays, lists, tuples and dicts.
    """

    @abc.abstractmethod
    def __getitem__(self, item: KeyT) -> ValT:
        pass

    @classmethod
    def __subclasshook__(cls, subclass):
        required = ['__getitem__', '__len__']
        compatible = (
                cls is Index and
                all(attr in subclass.__dict__ for attr in required)
        )
        return compatible or NotImplemented


class Executor(metaclass=abc.ABCMeta):
    """
    An abstract base class representing the executor API. An executor is any
    object that implements the `map` function.
    """

    @abc.abstractmethod
    def map(self,
            func: Callable[[Individual], Ord],
            individuals: Iterable[Individual]) -> Iterable[Ord]:
        pass

    @classmethod
    def __subclasshook__(cls, subclass):
        compatible = cls is Executor and 'map' in subclass.__dict__
        return compatible or NotImplemented


class Estimator(metaclass=abc.ABCMeta):
    """
    An abstract base class representing an abstract fitness estimator. The most
    basic estimator is simply a function, but it can be any Callable object with
    a compatible call signature.
    """

    @abc.abstractmethod
    def __call__(self,
                 individuals: Index[KeyT, Individual],
                 **kwargs) -> Index[KeyT, Ord]:
        pass


class Recorder(metaclass=abc.ABCMeta):
    """
    An abstract base class representing a record keeper. Recorders are
    responsible for starting and updating metadata records (e.g. fitness,
    age, etc.) associated with individuals.
    """

    @abc.abstractmethod
    def start(self,
              individuals: Index[KeyT, Individual],
              scores: Index[KeyT, Ord],
              *args, **kwargs) -> Index[KeyT, Record]:
        """
        Start records
        :param individuals:
        :param scores:
        :param kwargs:
        :return:
        """
        pass

    @abc.abstractmethod
    def update(self,
               individuals: Index[KeyT, Individual],
               records: Index[KeyT, Record],
               *args, **kwargs) -> Index[KeyT, Record]:
        """
        Update records
        :param individuals: all individuals at the start of a generation, i.e.
        all individuals that might've mated
        :param records: records associated with `individuals`
        :param kwargs:
        :return: an index of updated records
        """
        pass


class MateSelector(metaclass=abc.ABCMeta):

    """
    An abstract base class representing the a parent selection operator.
    """

    @property
    @abc.abstractmethod
    def nmates(self) -> int:
        """
        How many individuals form a mating group? This is required to check
        compatibility with a crossover operator
        :return:
        """
        pass

    @abc.abstractmethod
    def __call__(self,
                 npairs: int,
                 individuals: Index[KeyT, Individual],
                 records: Index[KeyT, Record],
                 **kwargs) -> Collection[Collection[KeyT]]:
        pass


class Crossover(metaclass=abc.ABCMeta):
    """
    An abstract base class representing a crossover operator
    """

    @property
    @abc.abstractmethod
    def nmates(self) -> int:
        """
        How many individuals form a crossover group? It's required to check
        compatibility with a parent selection operator
        :return:
        """
        pass

    @property
    @abc.abstractmethod
    def broodsize(self) -> int:
        """
        How many offsprings are produced by the crossover operation. It's
        required to calculate the number of mating groups to select.
        :return:
        """
        pass

    @abc.abstractmethod
    def __call__(self,
                 individuals: Index[KeyT, Individual],
                 records: Index[KeyT, Record],
                 mates: Index[KeyT, Collection[KeyT]],
                 **kwargs) -> Index[KeyT, Collection[Individual]]:
        """
        Carry out crossover on mating groups
        :param individuals:
        :param records:
        :param mates:
        :param kwargs:
        :return: an index of offspring collections aligned with mating groups,
        i.e. offsprings[i] are produced by mates[i];
        len(offspring[i]) == broodsize for any i in individuals
        """
        pass


class Mutator(metaclass=abc.ABCMeta):
    """
    An abstract base class representing a mutation operator
    """

    @abc.abstractmethod
    def __call__(self,
                 individuals: Index[KeyT, Individual],
                 **kwargs) -> Index[KeyT, Individual]:
        pass


class SelectionPolicy(metaclass=abc.ABCMeta):
    """
    An abstract base class representing a population selection operator
    """

    @abc.abstractmethod
    def __call__(self,
                 size: int,
                 individuals: Index[KeyT, Individual],
                 records: Index[KeyT, Record],
                 **kwargs) -> Collection[KeyT]:
        pass


# Join = Callable[
#     [Index[KeyType, ValType], Index[KeyType, ValType]],
#     Index[KeyType, ValType]
# ]


# Move all repetitive stuff to configurations
Operators = NamedTuple('Operators', [
    ('recorder', Recorder),
    ('estimator', Estimator),
    ('selector', MateSelector),
    ('crossover', Crossover),
    ('mutator', Mutator),
    # ('join', Join),
    ('policy', SelectionPolicy),
])


class Callback(metaclass=abc.ABCMeta):
    """
    An abstract base class representing a callback operation.
    """

    @abc.abstractmethod
    def __call__(
        self,
        individuals: Index[KeyT, Individual],
        records: Index[KeyT, Record],
        operators: Operators
    ) -> Tuple[Index[KeyT, Individual], Index[KeyT, Record], Operators]:
        pass


class BaseEvolver(metaclass=abc.ABCMeta):
    """
    An base Evolver class. It is responsible for orchestrating the evolution
    process and calling callbacks along the way
    """

    @abc.abstractmethod
    def evolve_generation(
        self,
        operators: Operators,
        gensize: int,  # the number of children per generation
        individuals: Index[KeyT, Individual],
        records: Optional[Index[KeyT, Record]] = None,
        **kwargs
    ) -> Tuple[Index[KeyT, Individual], Index[KeyT, Record]]:
        pass

    def evolve(
        self,
        generations: int,
        operators: Operators,
        gensize: int,
        individuals: Index[KeyT, Individual],
        records: Optional[Index[KeyT, Record]] = None,
        callbacks: Optional[List[Callback]] = None,
        **kwargs
    ) -> Tuple[Index[KeyT, Individual], Index[KeyT, Record]]:
        if not (isinstance(generations, int) and generations > 0):
            raise ValueError('generations must be a positive integer')
        for _ in range(generations):
            individuals, records, operators = self.call_callbacks(
                callbacks or [],
                *self.evolve_generation(
                    operators,
                    gensize,
                    individuals,
                    records,
                    **kwargs
                ),
                operators
            )
        return individuals, records  # type: ignore

    def call_callbacks(
        self,
        callbacks: List[Callback],
        individuals: Index[KeyT, Individual],
        records: Index[KeyT, Record],
        operators: Operators
    ) -> Tuple[Index[KeyT, Individual], Index[KeyT, Record], Operators]:
        return reduce(
            lambda arguments, callback: callback(*arguments),
            callbacks,
            (individuals, records, operators)
        )


if __name__ == '__main__':
    raise RuntimeError
