import random


__all__ = ["bimodal"]


def bimodal(fittest_fraction, other_random_survival):
    """
    :type fittest_fraction: float
    :param fittest_fraction:
    :type other_random_survival: float
    :param other_random_survival:
    """

    def bimodal_(sorted_population):
        """
        :type sorted_population: list[(Any, BaseIndividual)]
        :param sorted_population:
        :rtype: list[Any, BaseIndividual)]
        """
        # pick the most fittest and random lesser fit individuals
        n_fittest = int(len(sorted_population) * fittest_fraction)
        n_random = int((len(sorted_population) - n_fittest) * other_random_survival)
        fittest_survivors = sorted_population[:n_fittest]
        random_survivors = random.sample(sorted_population[n_fittest:], n_random)
        return fittest_survivors + random_survivors

    return bimodal_


if __name__ == "__main__":
    raise RuntimeError
