from swarm_algorithms.pso import PSO
from swarm_algorithms.spp import SPP
from swarm_algorithms.aaa import AAA
from testing_functions.auckley import Auckley
from testing_functions.michalkiewicz import Michalkiewicz
from tqdm import trange
import numpy as np

NUMBER_OF_STEPS = 30

NUMBER_OF_AGENTS = 50

OPTIMIZED_FUNCTON = Auckley()
#OPTIMIZED_FUNCTON = Michalkiewicz()

if __name__ == '__main__':
    swarm = AAA(optimized_function=OPTIMIZED_FUNCTON, number_of_agents=NUMBER_OF_AGENTS)

    particles_in_step = []

    for i in trange(NUMBER_OF_STEPS):
        swarm.step()
        if i % 3 == 0:
            print(swarm.optimized_function(swarm.best_solution), swarm.best_solution)
            OPTIMIZED_FUNCTON.plot_2d(points=np.array([particle.position for particle in swarm.particles]), dirs=np.array([particle.velocity for particle in swarm.particles]))

    OPTIMIZED_FUNCTON.plot_3d(points=np.array([particle.position for particle in swarm.particles]))
