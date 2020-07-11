from swarm_algorithms.pso import PSO
from swarm_algorithms.spp import SPP
from swarm_algorithms.aaa import AAA
from testing_functions.auckley import Auckley
from testing_functions.michalkiewicz import Michalkiewicz
from tqdm import trange
import numpy as np

NUMBER_OF_STEPS = 500
NUMBER_OF_AGENTS = 30

optimised_function = Auckley()
# optimised_function = Michalkiewicz()

if __name__ == '__main__':
    swarm = PSO(optimised_function=optimised_function, number_of_agents=NUMBER_OF_AGENTS)
    # swarm = SPP(optimised_function=optimised_function, number_of_agents=NUMBER_OF_AGENTS)
    # swarm = AAA(optimised_function=optimised_function, number_of_agents=NUMBER_OF_AGENTS)

    particles_in_step = []

    for i in trange(NUMBER_OF_STEPS):
        swarm.step()
        if i % 30 == 0:
            print(swarm.optimized_function(swarm.best_solution), swarm.best_solution)
            optimised_function.plot_2d(points=np.array([particle.position for particle in swarm.particles]), dirs=np.array([particle.velocity for particle in swarm.particles]))

    optimised_function.plot_3d(points=np.array([particle.position for particle in swarm.particles]))
