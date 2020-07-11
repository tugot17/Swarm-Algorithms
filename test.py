from swarm_algorithms.pso import PSO
from swarm_algorithms.spp import SPP
from swarm_algorithms.aaa import AAA
from testing_functions.auckley import Auckley
from testing_functions.michalkiewicz import Michalkiewicz
from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt

NUMBER_OF_STEPS = 1000

NUMBER_OF_AGENTS = 20

OPTIMIZED_FUNCTON = Auckley()
# OPTIMIZED_FUNCTON = Michalkiewicz()

if __name__ == '__main__':


    new_ratios = [i/20 for i in range(1, 22)]

    local_ratios = [i/20 for i in range(1, 22)]

    bests = np.zeros((len(new_ratios), len(local_ratios)))

    for idx, local_ratio in enumerate(local_ratios):
        for jdx, new_ratio in enumerate(new_ratios):
            swarm = PSO(optimised_function=OPTIMIZED_FUNCTON, number_of_agents=NUMBER_OF_AGENTS, new_ratio=new_ratio,
                        local_ratio=local_ratio)

            particles_in_step = []

            for i in trange(NUMBER_OF_STEPS):
                swarm.step()


            bests[idx,jdx] = OPTIMIZED_FUNCTON(swarm.best_solution)

    # Plot the surface.
    pl = plt.contourf(new_ratios, local_ratios, bests)
    plt.colorbar(pl, shrink=0.5, aspect=5)
    plt.show()
