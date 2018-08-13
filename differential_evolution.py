import numpy as np


class optimizer:
    def __init__(self, epoch, iterationCount, solutionSize, populationSize, F, CR, display, env):
        self.epoch = epoch
        self.iterationCount = iterationCount
        self.solutionSize = solutionSize
        self.populationSize = populationSize
        self.F = F
        self.CR = CR
        self.display = display
        self.env = env

    def optimize(self):

        errors = []
        bests = []
        losses = []

        for e in np.arange(self.epoch):

            population = np.random.random((self.populationSize, self.solutionSize))

            loss = []

            for iter in np.arange(self.iterationCount):
                solutionScores = []

                for idx in np.arange(self.populationSize):
                    solution = population[idx]
                    newSolution = []

                    while True:
                        indexes = np.random.choice(self.populationSize, 3, replace=False)
                        if idx not in indexes:
                            break

                    a = population[indexes[0]]
                    b = population[indexes[1]]
                    c = population[indexes[2]]

                    R = np.random.choice(self.populationSize)

                    for i in np.arange(len(solution)):
                        ri = np.random.random()
                        
                        if ri < self.CR or i == R:
                            newSolution.append(a[i] + self.F * (b[i] - c[i]))
                        else:
                            newSolution.append(solution[i])

                    newFitness = self.env.calculateFitness(newSolution)
                    oldFitness = self.env.calculateFitness(solution)

                    if newFitness < oldFitness:
                        population[idx] = newSolution

                    solutionScores.append(self.env.calculateFitness(population[idx]))

                bestSolutionIdx = solutionScores.index(min(solutionScores))
                loss.append(solutionScores[bestSolutionIdx])

                if self.display > 0 and iter > 0 and iter % self.display == 0:
                    print('Iteration:', iter, ' - Loss: ',  solutionScores[bestSolutionIdx])
                    self.env.drawSolution(population[bestSolutionIdx])

            errors.append(self.env.calculateError(population[bestSolutionIdx]))
            bests.append(population[bestSolutionIdx])
            losses.append(loss)

        sums = [sum(error) for error in errors]
        idx = np.asarray(sums).argsort()[0]
        bestIndividual = bests[idx]
        bestIndividualLoss = losses[idx]

        if self.display > 0:
            print('Average of errors: ', np.average(sums))

        return(sums, bests, losses, bestIndividual, bestIndividualLoss)