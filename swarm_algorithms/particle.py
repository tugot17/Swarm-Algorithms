import numpy as np


class Particle:
    def __init__(self, position, velocity=np.zeros(2)):
        self.position = position
        self.velocity = velocity
        self.best_known_position = np.copy(position)
        self.best_score = 0.