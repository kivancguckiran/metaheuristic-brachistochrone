import time
import numpy as np
import matplotlib.pyplot as plt
import util
import artificial_bee_colony
import differential_evolution
import genetic_algorithm
import harmony_search
import simulated_annealing


env = util.mediums(500, 500, 15)

abc_optimizer = artificial_bee_colony.optimizer(1, 4500, 15, 50, 50, 20, 20, 0, env)
de_optimizer = differential_evolution.optimizer(1, 1750, 15, 100, 0.5, 0.5, 0, env)
ga_optimizer = genetic_algorithm.optimizer(1, 6000, 15, 100, 40, 20, 0.1, 0, env)
hs_optimizer = harmony_search.optimizer(1, 5500, 15, 100, 0.99, 0.05, 0.005, 0, env)
sa_optimizer = simulated_annealing.optimizer(1, 15, 1.0, 0.00001, 0.99625, 100, 0, env)

startTime = time.time() * 1000
(sums, bests, losses, bestSolution, bestSolutionLoss) = abc_optimizer.optimize()
endTime = time.time() * 1000
print('ABC: Time spent: ', endTime - startTime, ' Error: ', bestSolutionLoss)

startTime = time.time() * 1000
(sums, bests, losses, bestSolution, bestSolutionLoss) = de_optimizer.optimize()
endTime = time.time() * 1000
print('DE: Time spent: ', endTime - startTime, ' Error: ', bestSolutionLoss)

startTime = time.time() * 1000
(sums, bests, losses, bestSolution, bestSolutionLoss) = ga_optimizer.optimize()
endTime = time.time() * 1000
print('GA: Time spent: ', endTime - startTime, ' Error: ', bestSolutionLoss)

startTime = time.time() * 1000
(sums, bests, losses, bestSolution, bestSolutionLoss) = hs_optimizer.optimize()
endTime = time.time() * 1000
print('HS: Time spent: ', endTime - startTime, ' Error: ', bestSolutionLoss)

startTime = time.time() * 1000
(sums, bests, losses, bestSolution, bestSolutionLoss) = sa_optimizer.optimize()
endTime = time.time() * 1000
print('SA: Time spent: ', endTime - startTime, ' Error: ', bestSolutionLoss)
