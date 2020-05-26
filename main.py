from swarm_algorithms.pso import PSO
from testing_functions.auckley import Auckley
from testing_functions.michalkiewicz import Michalkiewicz
from tqdm import trange

NUMBER_OF_STEPS = 2000

NUMBER_OF_AGENTS = 2000

OPTIMIZED_FUNCTON = Auckley()
# OPTIMIZED_FUNCTON = Michalkiewicz()

if __name__ == '__main__':
    # swarm = PSO(optimized_function=OPTIMIZED_FUNCTON, number_of_agents=NUMBER_OF_AGENTS)

    particles_in_step = []

    OPTIMIZED_FUNCTON.plot_3d()

    # for i in trange(NUMBER_OF_STEPS):
    #     swarm.step()
    #     particles_in_step.append(swarm.get_particles())




