import numpy as np
import matplotlib.pyplot as plt
import util
import artificial_bee_colony
import differential_evolution
import genetic_algorithm
import harmony_search
import simulated_annealing


env = util.mediums(500, 500, 15)

abc_optimizer = artificial_bee_colony.optimizer(1, 1000, 15, 50, 50, 20, 20, 10, env)
de_optimizer = differential_evolution.optimizer(1, 1000, 15, 100, 0.5, 0.5, 10, env)
ga_optimizer = genetic_algorithm.optimizer(1, 1000, 15, 100, 30, 40, 0.1, 10, env)
hs_optimizer = harmony_search.optimizer(1, 1000, 15, 100, 0.99, 0.05, 0.005, 10, env)
sa_optimizer = simulated_annealing.optimizer(1, 15, 1.0, 0.00001, 0.98, 100, 10, env)

# (sums, bests, losses, bestFoodSource, bestFoodSourceLoss) = abc_optimizer.optimize()
# print(bestFoodSource * 500)

# (sums, bests, losses, bestIndividual, bestIndividualLoss) = de_optimizer.optimize()
# print(bestIndividual * 500)

# (sums, bests, losses, bestIndividual, bestIndividualLoss) = ga_optimizer.optimize()
# print(bestIndividual * 500)

# (sums, bests, losses, bestMelody, bestMelodyLoss) = hs_optimizer.optimize()
# print(bestMelody * 500)

(sums, bests, losses, bestSolution, bestSolutionLoss) = sa_optimizer.optimize()
print(bestSolution * 500)
