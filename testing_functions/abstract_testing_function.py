from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


class AbstractTestingFunction(ABC):

    def __init__(self, **kwargs):
        super().__init__()


    def __call__(self, x):
        return NotImplementedError()

    @abstractmethod
    def plot_2d(self, points=None):
        return NotImplementedError()

    def plot_3d(self, points=None):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        X = np.arange(-40, 40, 0.5)
        Y = np.arange(-40, 40, 0.5)

        XY_plate = np.transpose([np.tile(X, len(Y)), np.repeat(Y, len(X))])

        print(f"{XY_plate.shape}")

        Z = self.__call__(XY_plate)

        Z = np.reshape(Z, (len(X), len(Y)))

        X, Y = np.meshgrid(X, Y)

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()
