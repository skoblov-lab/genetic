import unittest
import random

import numpy as np
import scipy.stats

from genetic import individuals, recombination, selection, populations


#TODO test recombination and selection


def trains(errors):
    """
    Check whether training reduces the error rate
    """
    test = scipy.stats.spearmanr(np.array(list(enumerate(errors))))
    return (test.correlation < 0) and (test.pvalue < 0.5)



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
        self.indiv = individuals.SingleChromosomeIndividual(self.engine, 0.1, 9)
        self.select = selection.bimodal(0.2, 0.05)
        self.nlegends = 10

    def test_single_thread(self):
        ancestors = [self.indiv] * 2
        pop = populations.PanmicticPopulation(ancestors, self.size,
                                              self.fitness, self.select,
                                              self.nlegends)


if __name__ == "__main__":
    unittest.main()
