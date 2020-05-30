import numpy as np
from testing_functions.abstract_testing_function import AbstractTestingFunction


class Michalkiewicz(AbstractTestingFunction):

    def __init__(self, **kwargs):
        self.M = kwargs.get('M', 2)

    def __call__(self, x):
        """
        x: vector of input values
        """

        d = x.shape[-1]

        return -np.sum(np.sin(x) * (np.sin(np.ones_like(x) * np.arange(1., d + 1.) * (x ** 2) / np.pi) ** (2 * self.M)), axis=-1)
