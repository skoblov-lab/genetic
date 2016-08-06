### genetic

_(ver. 0.1.dev2)_


This package is intended for numerical optimisation. The main goals are
flexibility and ease of use. A while ago I needed a genetic algorithms that
would allow to absolutely arbitrary combinations of objects (parameters) in the
genome. The lack of such implementation in Python (or my failure to find one)
pushed me towards developing this package. As of now it contains 4 main modules

- `individuals`
    - `BaseIndividual` - the base individual class. Any individual you'd wish to
    create must subclass this base class and implement all its methods and
    properties to make it compatible with any proper population. Refer to
    examples and docs for further details
    - `SingleChromosomeIndividual` – a basic individual that only has one
    chromosome. It is ageless. Mating is based on the `recombination.binomal`
    function.

- `populations`
todo
- `recombination` todo
- `selection` todo
- `util` todo

___

##### Example 1. Optimising sum



First let's create an individual. To do that we need an engine  that will
generate gene values

```
>>> import random

>>> def engine(_):
        return random.randint(0, 50)

```

Note that the engine need to have a single parameter. That is because in some
cases one may want to generate a new value based on the current state of the
genome. We will generate the values totally randomly.


Let's create the first individual. Let is have 10 genes. We can provide unique
engine for each gene, but in this case all genes will have the same engine.
And let's set the random mutation rate to `0.1`.

```
>>> from genetic.individuals import SingleChromosomeIndividual


>>> indiv = SingleChromosomeIndividual(engine, 0.1, 10)

```

Here we have out first individual.

Now, let's move on to the population. We will need to create the target (fitness)
function to maximise.
```


>>> def fitness(x):
        """
        :type x: SingleChromosomeIndividual
        """
        return - abs(200 - sum(x.genome))

```

So this function takes a `SingleChromosomeIndividual` instance and evaluates
the negative of the absolute difference between `200` and the sum of genes
(numbers in this case).

Now we need a selection model. Let's pick one from the `selection` module

```

>>> from genetic.selection import bimodal

>>> selection = bimodal(fittest_fraction=0.2, other_random_survival=0.05)

```

Here we used a `selection` factory, that has two parameters: the fraction
of the fittest individuals that survive a generation and the fraction of
random survivors in the rest of the population.

Now we can start the population. We need to pass at lest 2 individuals to begin
with, so we'll just take a copy of the same one. Let's have 100 individuals in
the population. And let's keep track of 10 legends

```
>>> from genetic.populations import PanmicticPopulation

>>> ancestors = [indiv] * 2
>>> population = PanmicticPopulation(ancestors, 100, fitness, selection, 10)

```

Now, let's make it evolve for 10 generations

```

>>> average_fitness = list(population.evolve(10))

```

Note that the `PanmicticPopulation.evolve` method returns a lazy generator, so
to make the evolution happen, you need to make it generate values (to iterate
over it). Our errors are:

```

>>> print(average_fitness)

[-59.979999999999997,
 -51.509999999999998,
 -41.18,
 -36.960000000000001,
 -30.359999999999999,
 -28.460000000000001,
 -28.82,
 -27.27,
 -28.5,
 -33.960000000000001]


```

Let's look at the legends' scores

```

>>> print([legend[0] for legend in population.legends])

[0, 0, 0, -1, -1, -1, -1, -1, -1, -1]


```

As you see, we already have 3 optimal solutions. Let's take a look at the first
one


```

>>> print(population.legends[0][1].genome)

(8, 31, 11, 8, 48, 2, 17, 25, 43, 7)

```

---

##### Example 2. Optimising neural network architecture

todo