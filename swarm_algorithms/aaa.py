from swarm_algorithms.abstract_swarm_algorithm import AbstractSwarmAlgorithm


class AAA(AbstractSwarmAlgorithm):

    def get_best_global_solution(self):
        return NotImplementedError()

    def step(self):
        return NotImplementedError()