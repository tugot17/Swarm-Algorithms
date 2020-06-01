import numpy as np


class Particle:
    def __init__(self, position, velocity=np.zeros(2), angle=0.):
        self.position = position
        self.velocity = velocity
        #self.angle = angle
        self.best_known_position = np.copy(position)
        self.best_score = 0.