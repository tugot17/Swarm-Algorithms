import numpy as np
from testing_functions.abstract_testing_function import AbstractTestingFunction


class Griewank(AbstractTestingFunction):

    def __call__(self, x):
        """
        x: vector of input values
        """
        d = x.shape[-1]

        i = np.array([np.sqrt(i+1) for i in range(d)])

        return np.sum(x**2/4000, axis=-1) - np.prod(np.cos(x/i), axis=-1) + 1

