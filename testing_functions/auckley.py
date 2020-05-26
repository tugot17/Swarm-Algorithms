import numpy as np
import matplotlib.pyplot as plt
from testing_functions.abstract_testing_function import AbstractTestingFunction
from matplotlib import cm


class Auckley(AbstractTestingFunction):

    def __init__(self, **kwargs):
        self.A = kwargs.get('A', 20)
        self.B = kwargs.get('B',0.2)
        self.C = kwargs.get('C', 2*np.pi)


    def __call__(self, x):
        """
        x: vector of input values
        """
        d = x.shape[-1]

        sum_sq_term = -self.A * np.exp(-self.B * np.sqrt(np.sum(x * x, axis=-1) / d))

        cos_term = -np.exp(np.sum(np.cos(self.C * x) / d, axis=-1))

        return self.A + np.exp(1) + sum_sq_term + cos_term

    def plot_2d(self, points=None):
        x = np.arange(-32, 33)
        y = [self.__call__(np.array([x_i])) for x_i in x]
        plt.plot(x, y)
        plt.show()





