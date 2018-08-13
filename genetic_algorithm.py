import numpy as np


class optimizer:
    def __init__(self, epoch, generationCount, geneSize, populationSize, breederCount, luckyCount, mutationChance, display, env):
        self.epoch = epoch
        self.generationCount = generationCount
        self.geneSize = geneSize
        self.populationSize = populationSize
        self.breederCount = breederCount
        self.luckyCount = luckyCount
        self.mutationChance = mutationChance
        self.display = display
        self.env = env

    def optimize(self):
        errors = []
        bests = []
        losses = []

        for e in np.arange(self.epoch):
            population = np.random.random((self.populationSize, self.geneSize))

            loss = []

            for iter in np.arange(self.generationCount):
                fitnessScores = []

                for individual in population:
                    fitnessScores.append(self.env.calculateFitness(individual))

                bestIndexes = np.asarray(fitnessScores).argsort()
                bestIndividual = population[bestIndexes[0]]
                bestIndividualScore = fitnessScores[bestIndexes[0]]
                breederIndexes = bestIndexes[:self.breederCount]
                breederIndexes = np.reshape(breederIndexes, (int(self.breederCount / 2), 2))
                luckyIndexes = bestIndexes[self.breederCount:self.breederCount + self.luckyCount]

                loss.append(bestIndividualScore)

                if self.display > 0 and iter > 0 and iter % self.display == 0:
                    print('Iteration:', iter, ' - Loss: ',  bestIndividualScore)
                    self.env.drawSolution(bestIndividual)

                newPopulation = []

                for index in breederIndexes:
                    father = population[index[0]]
                    mother = population[index[1]]

                    pos = int(np.random.random() * self.geneSize)

                    child1 = np.concatenate((father[:pos], mother[pos:]), axis=0)
                    child2 = np.concatenate((father[pos:], mother[:pos]), axis=0)

                    for i in np.arange(len(child1)):
                        if self.mutationChance > np.random.random():
                            child1[i] = np.random.random()
                    for i in np.arange(len(child2)):
                        if self.mutationChance > np.random.random():
                            child2[i] = np.random.random()

                    newPopulation.append(father)
                    newPopulation.append(mother)
                    newPopulation.append(child1)
                    newPopulation.append(child2)

                population = np.concatenate((newPopulation, population[luckyIndexes]), axis=0)

                np.random.shuffle(population)

            errors.append(self.env.calculateError(bestIndividual))
            bests.append(bestIndividual)
            losses.append(loss)

        sums = [sum(error) for error in errors]
        idx = np.asarray(sums).argsort()[0]
        bestIndividual = bests[idx]
        bestIndividualLoss = losses[idx]

        if self.display > 0:
            print('Average of errors: ', np.average(sums))

        return(sums, bests, losses, bestIndividual, bestIndividualLoss)
