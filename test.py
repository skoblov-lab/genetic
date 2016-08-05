import unittest
import random

import numpy as np
import scipy.stats

from genetic import individuals, recombination, selection, populations


#TODO test recombination and selection


class TestSingleChromosomeIndividual(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = lambda x: random.randint(0, 50)
        self.mutrate = 0.1

    def test_mutrate(self):
        with self.assertRaises(ValueError):
            individuals.SingleChromosomeIndividual(self.engine,
                                                   1.2, l=10)
        with self.assertRaises(ValueError):
            individuals.SingleChromosomeIndividual(self.engine,
                                                   -0.1, l=10)
        with self.assertRaises(ValueError):
            individuals.SingleChromosomeIndividual(self.engine,
                                                   1, l=10)
        individuals.SingleChromosomeIndividual(self.engine, self.mutrate, l=10)

    def test_engine_and_length(self):
        with self.assertRaises(ValueError):
            individuals.SingleChromosomeIndividual([self.engine],
                                                   self.mutrate, l=10)
        with self.assertRaises(ValueError):
            individuals.SingleChromosomeIndividual(self.engine, self.mutrate)
        with self.assertRaises(ValueError):
            individuals.SingleChromosomeIndividual([self.engine, 1],
                                                   self.mutrate)
        individuals.SingleChromosomeIndividual(self.engine, self.mutrate, l=10)
        individuals.SingleChromosomeIndividual([self.engine], self.mutrate, l=1)


class TestPanmicticPopulation(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = lambda x: random.randint(0, 50)
        self.fitness = lambda x: - abs(200 - sum(x.genome))
        self.size = 100
        self.indiv = individuals.SingleChromosomeIndividual(self.engine, 0.1, 50)
        self.select = selection.bimodal(0.2, 0.05)
        self.nlegends = 10

    def test_single_thread(self):
        ancestors = [self.indiv] * 2
        pop = populations.PanmicticPopulation(ancestors, self.size,
                                              self.fitness, self.select,
                                              self.nlegends)
        errors1 = list(map(abs, pop.evolve(5, jobs=1)))
        errors2 = list(map(abs, pop.evolve(100, jobs=1)))

        self.assertTrue(np.mean(errors2) < np.mean(errors1))

    def test_two_threads(self):
        ancestors = [self.indiv] * 2
        pop = populations.PanmicticPopulation(ancestors, self.size,
                                              self.fitness, self.select,
                                              self.nlegends)
        errors1 = list(map(abs, pop.evolve(5, jobs=1)))
        errors2 = list(map(abs, pop.evolve(100, jobs=1)))
        self.assertTrue(np.mean(errors2) < np.mean(errors1))

    def test_legends(self):
        ancestors = [self.indiv] * 2
        with self.assertRaises(ValueError):
            populations.PanmicticPopulation(ancestors, self.size,
                                            self.fitness, self.select,
                                            -10)
        with self.assertRaises(ValueError):
            populations.PanmicticPopulation(ancestors, self.size,
                                            self.fitness, self.select,
                                            0.1)

        pop = populations.PanmicticPopulation(ancestors, self.size,
                                              self.fitness, self.select,
                                              self.nlegends)
        errors1 = list(map(abs, pop.evolve(1, jobs=1)))
        legends1 = pop.legends
        errors2 = list(map(abs, pop.evolve(49, jobs=1)))
        legends2 = pop.legends

        legendary_scores1 = [abs(legend[0]) for legend in legends1]
        legendary_scores2 = [abs(legend[0]) for legend in legends2]

        self.assertTrue(np.mean(legendary_scores2) < np.mean(legendary_scores1))


if __name__ == "__main__":
    unittest.main()
