from swarm_algorithms.abstract_swarm_algorithm import AbstractSwarmAlgorithm
import numpy as np

from swarm_algorithms.particle import Particle


class AAA(AbstractSwarmAlgorithm):
    # Hyperparameters
    w_min, w_max = np.pi / 2, 3 * np.pi / 2

    a_min, a_max = 0., 2 * np.pi
    p_lo, p_hi = -15., 15.

    survival_coeff = 0.7
    s_sigma = 0.01


    def __init__(self, optimized_function, number_of_agents):
        super().__init__(optimized_function, number_of_agents)

        # Initialize particles
        self.particles = [Particle(position=np.random.uniform(self.p_lo, self.p_hi, 2))
                          for _ in range(self.number_of_agents)]

        # Update the swarm best known position
        self.best_swarm_score = np.inf
        for particle in self.particles:
            particle.best_score = self.optimized_function(particle.position)
            if particle.best_score < self.best_swarm_score:
                self.best_swarm_score = particle.best_score
                self.best_solution = particle.best_known_position

    def get_best_global_solution(self):
        return self.best_solution

    def step(self):
        r_max = np.mean([np.sqrt((p1.position - p2.position) ** 2) for p1 in self.particles for p2 in self.particles])/2.

        w = np.random.uniform(self.w_min, self.w_max)
        for particle in self.particles:
            a = np.random.uniform(self.a_min, self.a_max)
            r = np.random.uniform(0., r_max)

            v1 = np.array([np.cos(a), np.sin(a)]) * r
            v2 = np.array([np.cos(w + a - np.pi), np.sin(w + a - np.pi)]) * r

            particle.position = particle.position + v1 + v2

        #starvation
        energy = [(particle, -self.optimized_function(particle.position)) for particle in self.particles]
        energy = list(sorted(energy, key=lambda x: -x[1]))
        survived = [p for p, _ in energy[:int(self.survival_coeff * len(energy))]]

        self.particles = survived + [Particle(survived[i].position + np.random.normal(0., self.s_sigma, survived[i].position.shape))
                                     for i in range(len(self.particles) - len(survived))]

        for particle in self.particles:
            particle_new_score = self.optimized_function(particle.position)
            if particle_new_score <= particle.best_score:
                particle.best_score = particle_new_score
                particle.best_known_position = np.copy(particle.position)
                if particle.best_score <= self.best_swarm_score:
                    self.best_swarm_score = particle.best_score
                    self.best_solution = particle.best_known_position