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

    def plot_3d(self, points=None):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        X = np.arange(0, 4, 0.1)
        Y = np.arange(0, 4, 0.1)

        XY_plate = np.transpose([np.tile(X, len(Y)), np.repeat(Y, len(X))])

        print(f"{XY_plate.shape}")

        Z = self.__call__(XY_plate)

        Z = np.reshape(Z, (len(X),len(Y)))

        X, Y = np.meshgrid(X, Y)


        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()
