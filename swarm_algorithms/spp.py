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
    def __init__(self, optimised_function, number_of_agents):
        super().__init__(optimised_function, number_of_agents)

        # Hyperparameters
        p_lo, p_hi = -15., 15.
        self.v = 0.05
        self.distance_metric = lambda a, b: np.sqrt(np.sum((a - b) ** 2))
        self.k_nearest = max(1, int(0.1 * number_of_agents))
        self.v_sigma = 0.001
        self.gamma = 0.1


        # Initialize particles
        velocity = np.random.uniform(-1., 1., (self.number_of_agents, 2))
        self.particles = [Particle(position=np.random.uniform(p_lo, p_hi, 2),
                                   velocity=self.v * velocity[i] / np.linalg.norm(velocity[i])
                                   )
                          for i in range(self.number_of_agents)]

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
        for particle in self.particles:
            dir_vector = self.best_solution - particle.position
            particle.velocity = self.v * dir_vector / np.linalg.norm(dir_vector) if np.sum(dir_vector) != 0. else dir_vector

        for particle in self.particles:
            neighbours = [(neighbour, self.distance_metric(particle.position, neighbour.position))
                          for neighbour in self.particles if neighbour is not particle]
            neighbours = list(sorted(neighbours, key=lambda x: x[1]))[:self.k_nearest]

            particle.velocity = (1. - self.gamma) * particle.velocity + \
                                self.gamma * np.mean(np.array([n.velocity for n, _ in neighbours]), axis=0) + \
                                np.random.normal(0., self.v_sigma, (2, ))

            particle.velocity = self.v * particle.velocity / np.linalg.norm(particle.velocity)

            particle.position = particle.position + particle.velocity

        for particle in self.particles:
            particle_new_score = self.optimized_function(particle.position)
            if particle_new_score < particle.best_score:
                particle.best_score = particle_new_score
                particle.best_known_position = np.copy(particle.position)
                if particle.best_score < self.best_swarm_score:
                    self.best_swarm_score = particle.best_score
                    self.best_solution = particle.best_known_position