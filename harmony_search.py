import numpy as np


class optimizer:
    def __init__(self, epoch, iterationCount, solutionSize, harmonyMemorySize, harmonyMemoryConsiderationRate, pitchAdjustRate, bandwidth, display, env):
        self.epoch = epoch
        self.iterationCount = iterationCount
        self.solutionSize = solutionSize
        self.harmonyMemorySize = harmonyMemorySize
        self.harmonyMemoryConsiderationRate = harmonyMemoryConsiderationRate
        self.pitchAdjustRate = pitchAdjustRate
        self.bandwidth = bandwidth
        self.display = display
        self.env = env

    def optimize(self):

        errors = []
        bests = []
        losses = []

        for e in np.arange(self.epoch):

            harmonyMemory = np.random.random((self.harmonyMemorySize, self.solutionSize))

            loss = []

            for iter in np.arange(self.iterationCount):
                solutionScores = []

                for idx in np.arange(self.harmonyMemorySize):
                    solution = harmonyMemory[idx]

                    for i in np.arange(len(solution)):
                        if np.random.rand() < self.harmonyMemoryConsiderationRate:
                            if np.random.rand() < self.pitchAdjustRate:
                                change = self.bandwidth * np.random.rand()
                                if np.random.rand() < 0.5:
                                    solution[i] += change
                                else:
                                    solution[i] -= change

                                if solution[i] < 0:
                                    solution[i] = 0
                                if solution[i] > 1:
                                    solution[i] = 1
                        else:
                            solution[i] = np.random.rand()

                    solutionScores.append(self.env.calculateFitness(solution))

                bestSolutionIdx = solutionScores.index(min(solutionScores))
                worstSolutionIdx = solutionScores.index(max(solutionScores))

                harmonyMemory[worstSolutionIdx] = np.copy(harmonyMemory[bestSolutionIdx])

                loss.append(solutionScores[bestSolutionIdx])

                if self.display > 0 and iter > 0 and iter % self.display == 0:
                    print('Iteration:', iter, ' - Loss: ',  solutionScores[bestSolutionIdx])
                    self.env.drawSolution(harmonyMemory[bestSolutionIdx])


            errors.append(self.env.calculateError(harmonyMemory[bestSolutionIdx]))
            bests.append(harmonyMemory[bestSolutionIdx])
            losses.append(self.env.calculateFitness(harmonyMemory[bestSolutionIdx]))


        sums = [sum(error) for error in errors]
        idx = np.asarray(sums).argsort()[0]
        bestMelody = bests[idx]
        bestMelodyLoss = losses[idx]

        if self.display > 0:
            print('Average of errors: ', np.average(sums))

        return(sums, bests, losses, bestMelody, bestMelodyLoss)
