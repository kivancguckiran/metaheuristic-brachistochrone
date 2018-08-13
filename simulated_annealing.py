import numpy as np


class optimizer:
    def __init__(self, epoch, solutionSize, maxTemperature, minTemperature, alpha, compare, display, env):
        self.epoch = epoch
        self.solutionSize = solutionSize
        self.maxTemperature = maxTemperature
        self.minTemperature = minTemperature
        self.alpha = alpha
        self.compare = compare
        self.display = display
        self.env = env

    def optimize(self):

        errors = []
        bests = []
        losses = []

        step = 1

        for e in np.arange(self.epoch):

            temperature = self.maxTemperature
            solution = np.random.random(self.solutionSize)

            loss = []

            step = 1
            iter = 0

            while temperature > self.minTemperature:

                for i in np.arange(self.compare):
                    pos = np.random.randint(self.solutionSize)
                    newSolution = np.copy(solution)
                    newSolution[pos] = np.random.rand()

                    oldCost = self.env.calculateFitness(solution)
                    newCost = self.env.calculateFitness(newSolution)

                    if np.random.rand() > np.exp((newCost - oldCost) * 100 / temperature):
                        solution = np.copy(newSolution)

                err = self.env.calculateFitness(solution)

                loss.append(err)

                if self.display > 0 and iter > 0 and iter % self.display == 0:
                    print('Iteration:', iter, ' - Loss: ',  err)
                    self.env.drawSolution(solution)

                iter += 1

                temperature = temperature * self.alpha
                step += 1

            errors.append(self.env.calculateError(solution))
            bests.append(solution)
            losses.append(loss)


        sums = [sum(error) for error in errors]
        idx = np.asarray(sums).argsort()[0]
        bestSolution = bests[idx]
        bestSolutionLoss = losses[idx]

        if self.display > 0:
            print('Average of errors: ', np.average(sums))

        return(sums, bests, losses, bestSolution, bestSolutionLoss)

