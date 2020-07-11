from swarm_algorithms.abstract_swarm_algorithm import AbstractSwarmAlgorithm
from swarm_algorithms.particle import Particle
import numpy as np


class PSO(AbstractSwarmAlgorithm):

    def __init__(self, optimised_function, number_of_agents, new_ratio=0.5, local_ratio = 0.5):
        """

        :param optimised_function:
        :param number_of_agents:
        :param new_ratio: how to affect new v
        :param local_ratio: how to affect local v
        """
        super().__init__(optimised_function, number_of_agents)
        self.w = 0.1 * (1 - new_ratio)
        self.fi_p = 0.1 * new_ratio * local_ratio
        self.fi_g = 0.1 * new_ratio * (1 - local_ratio)

        # Initialize particles
        p_lo, p_hi = -15., 15.
        v_lo, v_hi = -0.1, 0.1
        self.particles = [Particle(np.random.uniform(p_lo, p_hi, 2),
                                   np.random.uniform(v_lo, v_hi, 2)
                                   )
                          for i in range(self.number_of_agents)]

        #Update the swarm best known position
        self.best_swarm_score = np.inf
        for particle in self.particles:
            particle.best_score = self.optimized_function(particle.position)
            if particle.best_score < self.best_swarm_score:
                self.best_swarm_score = particle.best_score
                self.best_solution = particle.best_known_position

    def get_best_global_solution(self):
        return self.best_solution

    def step(self):
        """
        for each particle i = 1, ..., S do
          for each dimension d = 1, ..., n do
             Pick random numbers: rp, rg ~ U(0,1)
             Update the particle's velocity: vi,d ← ω vi,d + φp rp (pi,d-xi,d) + φg rg (gd-xi,d)
          Update the particle's position: xi ← xi + vi
          if f(xi) < f(pi) then
             Update the particle's best known position: pi ← xi
             if f(pi) < f(g) then
                Update the swarm's best known position: g ← pi
        """

        for particle in self.particles:
            rp = np.random.uniform(0., 1., particle.velocity.shape)
            rg = np.random.uniform(0., 1., particle.velocity.shape)
            particle.velocity = self.w * particle.velocity + \
                                self.fi_p * rp * (particle.best_known_position - particle.position) + \
                                self.fi_g * rg * (self.best_solution - particle.position)
            particle.position = particle.position + particle.velocity

        for particle in self.particles:
            particle_new_score = self.optimized_function(particle.position)
            if particle_new_score < particle.best_score:
                particle.best_score = particle_new_score
                particle.best_known_position = np.copy(particle.position)
                if particle.best_score < self.best_swarm_score:
                    self.best_swarm_score = particle.best_score
                    self.best_solution = particle.best_known_position

