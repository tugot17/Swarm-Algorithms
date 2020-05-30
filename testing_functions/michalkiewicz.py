import numpy as np
import matplotlib.pyplot as plt
from testing_functions.abstract_testing_function import AbstractTestingFunction
from matplotlib import cm


class Michalkiewicz(AbstractTestingFunction):

    def __init__(self, **kwargs):
        self.M = kwargs.get('M', 2)

    def __call__(self, x):
        """
        x: vector of input values
        """
        d = x.shape[-1]

        return -np.sum(np.sin(x) * (np.sin(np.arange(d) * (x ** 2) / np.pi) ** (2 * self.M)), axis=-1)


    def plot_2d(self, points=None):
        x = np.arange(-32, 33)
        y = [self.__call__(np.array([x_i])) for x_i in x]
        plt.plot(x, y)
        plt.show()


