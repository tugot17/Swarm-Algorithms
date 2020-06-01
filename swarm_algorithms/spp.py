from swarm_algorithms.abstract_swarm_algorithm import AbstractSwarmAlgorithm
import numpy as np
from swarm_algorithms.particle import Particle


class SPP(AbstractSwarmAlgorithm):
    """
    SPP are point particles, which
        - move with a constant speed
        - and adopt (at each time increment) the average direction of motion of the other particles
          in their local neighborhood
          up to some added noise.
    """
    def __init__(self):
        super().__init__(optimized_function, number_of_agents)

        # Initialize particles
        p_lo, p_hi = -50., 50.
        v = 1.0

        self.particles = [Particle(position=np.random.uniform(p_lo, p_hi, 2),
                                   angle=np.random.uniform(0., 2*np.pi),
                                   velocity=v
                                   )
                          for _ in range(self.number_of_agents)]


    def get_best_global_solution(self):
        return NotImplementedError()

    def step(self):
        return NotImplementedError()