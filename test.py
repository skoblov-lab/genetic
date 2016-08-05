import unittest
import random

from genetic import individuals, recombination, selection, populations


class TestIndividual(unittest.TestCase):
    pass


class TestPanmicticPopulation(unittest.TestCase):
    def test_all(self):
        def engine(_):
            return random.randint(0, 50)

        def fitness(x):
            return - abs(200 - sum(x.genome))

        pop_size = 200

        ancestors = [individuals.SingleChromosomeIndividual(engine, 0.1, 50)
                     for _ in range(pop_size)]

        select = selection.bimodal(0.2, 0.05)

        population = populations.PanmicticPopulation(ancestors, pop_size, fitness,
                                                     select, nlegends=10)
        print(list(population.evolve(50, 2)))
        print([ind[0] for ind in population.legends])


if __name__ == "__main__":
    unittest.main()
