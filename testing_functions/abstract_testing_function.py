from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


class AbstractTestingFunction(ABC):
    i=0

    def __init__(self, **kwargs):
        super().__init__()


    def __call__(self, x):
        return NotImplementedError()

    def plot_2d(self, points=[], dirs=[]):
        # Make data.
        dx_max = np.max(points[:, 0])
        dy_max = np.max(points[:, 1])

        dx_min = np.min(points[:, 0])
        dy_min = np.min(points[:, 1])

        d_min = min(dx_min, dy_min)
        d_max = max(dx_max, dy_max)

        d_min = min(-1, d_min)
        d_max = max(1, d_max)


        X = np.arange(d_min, d_max + (d_max - d_min) / 50., (d_max - d_min) / 50.)
        Y = np.arange(d_min, d_max + (d_max - d_min) / 50., (d_max - d_min) / 50.)

        XY_plate = np.transpose([np.tile(X, len(Y)), np.repeat(Y, len(X))])

        Z = self.__call__(XY_plate)
        Z = np.reshape(Z, (len(X), len(Y)))


        # Plot the surface.
        pl = plt.contourf(X, Y, Z)
        plt.colorbar(pl, shrink=0.5, aspect=5)

        # Plot the points
        plt.scatter(points[:, 0], points[:, 1], c='r')

        # Plot the dirs
        plt.quiver(points[:, 0], points[:, 1], dirs[:, 0], dirs[:, 1])

        plt.savefig(f"gifs/{self.i}.jpg")

        self.i += 1

        plt.show()

    def plot_3d(self, points=[], d=10):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        dx_max = np.max(points[:, 0])
        dy_max = np.max(points[:, 1])

        dx_min = np.min(points[:, 0])
        dy_min = np.min(points[:, 1])

        d_min = min(dx_min, dy_min)
        d_max = max(dx_max, dy_max)

        d_min = min(-1, d_min)
        d_max = max(1, d_max)


        X = np.arange(d_min, d_max + (d_max - d_min) / 50., (d_max - d_min) / 50.)
        Y = np.arange(d_min, d_max + (d_max - d_min) / 50., (d_max - d_min) / 50.)

        XY_plate = np.transpose([np.tile(X, len(Y)), np.repeat(Y, len(X))])

        Z = self.__call__(XY_plate)
        Z = np.reshape(Z, (len(X), len(Y)))
        X, Y = np.meshgrid(X, Y)

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        # Plot the points
        ax.scatter(points[:, 0], points[:, 1], self.__call__(points), c='g')

        plt.show()
