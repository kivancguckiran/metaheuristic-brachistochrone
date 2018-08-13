import numpy as np


def selectOne(probabilities):
    max     = np.sum(probabilities)
    pick    = np.random.uniform(0, max)
    current = 0
    idx     = 0
    for prob in probabilities:
        current += prob
        if current > pick:
            break
        idx += 1

    return idx

class optimizer:
    def __init__(self, epoch, iterationCount, foodSourceSize, foodSourceCount, limitForScout, employedBeeCount, onlookerBeeCount, display, env):
        self.epoch = epoch
        self.iterationCount = iterationCount
        self.foodSourceSize = foodSourceSize
        self.foodSourceCount = foodSourceCount
        self.limitForScout = limitForScout
        self.employedBeeCount = employedBeeCount
        self.onlookerBeeCount = onlookerBeeCount
        self.display = display
        self.env = env

    def optimize(self):

        losses = []
        bests = []
        errors = []

        for e in np.arange(self.epoch):

            foodSources = np.random.random((self.foodSourceCount, self.foodSourceSize))
            trialScores = np.zeros(self.foodSourceCount)

            loss = []

            for iter in np.arange(self.iterationCount):

                # Employed Bees
                for idx in np.arange(self.employedBeeCount):

                    i = np.random.randint(self.foodSourceCount)
                    k = np.random.randint(self.foodSourceCount)
                    j = np.random.randint(self.foodSourceSize)
                    fi = np.random.uniform(-1, 1)

                    newFoodSource = np.copy(foodSources[i])
                    alternateFoodSource = foodSources[k]
                    newFoodSource[j] = newFoodSource[j] + fi * (newFoodSource[j] - alternateFoodSource[j])

                    if newFoodSource[j] < 0:
                        newFoodSource[j] = 0
                    if newFoodSource[j] > 1:
                        newFoodSource[j] = 1

                    oldScore = self.env.calculateFitness(foodSources[i])
                    newScore = self.env.calculateFitness(newFoodSource)

                    if newScore < oldScore:
                        foodSources[i] = newFoodSource
                        trialScores[i] = 0
                    else:
                        trialScores[i] += 1

                fitnessScores = []

                fitnessScores = [self.env.calculateFitness(foodSource) for foodSource in foodSources]
                probabilities = (fitnessScores - np.min(fitnessScores)) / np.ptp(fitnessScores)
                probabilities = [1 - prob for prob in probabilities]


                # Onlooker Bees
                for idx in np.arange(self.onlookerBeeCount):

                    i = selectOne(probabilities)
                    k = np.random.randint(self.foodSourceCount)
                    j = np.random.randint(self.foodSourceSize)
                    fi = np.random.uniform(-1, 1)

                    newFoodSource = np.copy(foodSources[i])
                    alternateFoodSource = foodSources[k]
                    newFoodSource[j] = newFoodSource[j] + fi * (newFoodSource[j] - alternateFoodSource[j])

                    if newFoodSource[j] < 0:
                        newFoodSource[j] = 0
                    if newFoodSource[j] > 1:
                        newFoodSource[j] = 1

                    oldScore = self.env.calculateFitness(foodSources[i])
                    newScore = self.env.calculateFitness(newFoodSource)

                    if newScore < oldScore:
                        foodSources[i] = newFoodSource
                        trialScores[i] = 0
                    else:
                        trialScores[i] += 1

                # Scout Bees
                for idx in np.arange(len(trialScores)):
                    if trialScores[idx] > self.limitForScout:
                        foodSources[idx] = np.random.random(self.foodSourceSize)
                        trialScores[idx] = 0

                bestIndexes = np.asarray(fitnessScores).argsort()
                bestFoodSource = foodSources[bestIndexes[0]]

                loss.append(fitnessScores[0])

                if self.display > 0 and iter > 0 and iter % self.display == 0:
                    print('Iteration:', iter, ' - Loss: ',  fitnessScores[0])
                    self.env.drawSolution(bestFoodSource)

            errors.append(self.env.calculateError(bestFoodSource))
            bests.append(bestFoodSource)
            losses.append(loss)

        sums = [sum(error) for error in errors]
        idx = np.asarray(sums).argsort()[0]
        bestFoodSource = bests[idx]
        bestFoodSourceLoss = losses[idx]

        if self.display > 0:
            print('Average of errors: ', np.average(sums))

        return(sums, bests, losses, bestFoodSource, bestFoodSourceLoss)
