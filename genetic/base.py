import abc
from itertools import chain
from functools import reduce
from typing import Callable, Collection, Generic, List, NamedTuple, \
    Optional, Tuple, TypeVar, Iterable

KeyType = TypeVar('KeyType')
ValType = TypeVar('ValType', covariant=True)
Individual = TypeVar('Individual', covariant=True)
Record = TypeVar('Record', covariant=True)


class Ord(metaclass=abc.ABCMeta):

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


class Index(Generic[KeyType, ValType], metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __getitem__(self, item: KeyType) -> ValType:
        pass

    @classmethod
    def __subclasshook__(cls, subclass):
        required = ['__getitem__', '__contains__', '__len__']
        compatible = (
                cls is Index and
                all(attr in subclass.__dict__ for attr in required)
        )
        return compatible or NotImplemented


class Executor(metaclass=abc.ABCMeta):

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

    @abc.abstractmethod
    def __call__(self,
                 individuals: Index[KeyType, Individual],
                 **kwargs) -> Index[KeyType, Ord]:
        pass


class Recorder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def start(self,
              individuals: Index[KeyType, Individual],
              scores: Index[KeyType, Ord],
              **kwargs) -> Index[KeyType, Record]:
        pass

    @abc.abstractmethod
    def update(self,
               individuals: Index[KeyType, Individual],
               records: Index[KeyType, Record],
               mates: Collection[Collection[KeyType]],
               broodsize: int,
               **kwargs) -> Index[KeyType, Record]:
        """
        :param individuals: all individuals at the start of a generation, i.e.
        all individuals that might've mated
        :param records: records associated with `individuals`
        :param mates: mating groups
        :param broodsize: the number of children per mating group
        :param kwargs:
        :return:
        """
        pass


class MateSelector(metaclass=abc.ABCMeta):

    # TODO does it make sense to explicitly show how many mates are selected

    @property
    @abc.abstractmethod
    def nmates(self) -> int:
        """
        How many individuals form a mating group?
        :return:
        """
        pass

    @abc.abstractmethod
    def __call__(self,
                 npairs: int,
                 individuals: Index[KeyType, Individual],
                 records: Index[KeyType, Record],
                 **kwargs) -> Collection[Collection[KeyType]]:
        pass


class Crossover(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def nmates(self) -> int:
        """
        How many individuals form a crossover group?
        :return:
        """
        pass

    @property
    @abc.abstractmethod
    def broodsize(self) -> int:
        pass

    @abc.abstractmethod
    def __call__(self,
                 individuals: Index[KeyType, Individual],
                 records: Index[KeyType, Record],
                 mates: Collection[Collection[KeyType]],
                 **kwargs) -> Index[KeyType, Individual]:
        pass


class ChildMutator(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self,
                 individuals: Index[KeyType, Individual],
                 **kwargs) -> Index[KeyType, Individual]:
        pass


class SelectionPolicy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self,
                 size: int,
                 individuals: Index[KeyType, Individual],
                 records: Index[KeyType, Record],
                 **kwargs) -> Collection[KeyType]:
        pass


Join = Callable[
    [Index[KeyType, ValType], Index[KeyType, ValType]],
    Index[KeyType, ValType]
]


# Move all repetitive stuff to configurations
Operators = NamedTuple('Operators', [
    ('recorder', Recorder),
    ('estimator', Estimator),
    ('selector', MateSelector),
    ('crossover', Crossover),
    ('mutator', ChildMutator),
    ('join', Join),
    ('policy', SelectionPolicy),
])


class Callback(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(
        self,
        individuals: Index[KeyType, Individual],
        records: Index[KeyType, Record],
        operators: Operators
    ) -> Tuple[Index[KeyType, Individual], Index[KeyType, Record], Operators]:
        pass


class BaseEvolver(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def evolve_generation(
        self,
        operators: Operators,
        gensize: int,  # the number of children per generation
        individuals: Index[KeyType, Individual],
        records: Optional[Index[KeyType, Record]] = None,
        **kwargs
    ) -> Tuple[Index[KeyType, Individual], Index[KeyType, Record]]:
        pass

    def evolve(
        self,
        generations: int,
        operators: Operators,
        gensize: int,
        individuals: Index[KeyType, Individual],
        records: Optional[Index[KeyType, Record]] = None,
        callbacks: Optional[List[Callback]] = None,
        **kwargs
    ) -> Tuple[Index[KeyType, Individual], Index[KeyType, Record]]:
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
        individuals: Index[KeyType, Individual],
        records: Index[KeyType, Record],
        operators: Operators
    ) -> Tuple[Index[KeyType, Individual], Index[KeyType, Record], Operators]:
        return reduce(
            lambda arguments, callback: callback(*arguments),
            callbacks,
            (individuals, records, operators)
        )


if __name__ == '__main__':
    raise RuntimeError
