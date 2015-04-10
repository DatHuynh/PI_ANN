__author__ = 'DatHuynh'

from deap import base, creator,tools
import random
import numpy as np
import Network as nw
import math
import random

from itertools import repeat
from collections import Sequence

ranges = [[500,5000],[10,15],[30000,95000],[30,40],[42000,494000],[33,42],[42000,494000],[33,42],[42000,494000],[33,42],[42000,494000],[33,42],[42000,494000],[33,42],[42000,494000],[33,	42],[42000,	494000],[33,	42],[95000,	5600000],[18,	28]]

def initPs(container):
    min = 0
    max = 1
    Ps = container(np.zeros(len(ranges)))
    for i in range(0,len(ranges)):
        if i > 0 and i%2 != 0:
            low = (Ps[i-2]*(ranges[i-2][max]-ranges[i-2][min]) + ranges[i-2][min] - range[i][min])/(range[i][max]-range[i][min])
            if low > 0:
                Ps[i] = random.random()*(1.0-low)+low
            else:
                Ps[i] = random.random()
        else:
            Ps[i] = random.random()

def adjustConstraint(Ps):
    for i in range(0,len(ranges)):
        if i > 0 and i%2 != 0:
            low = (Ps[i-2]*(ranges[i-2][max]-ranges[i-2][min]) + ranges[i-2][min] - range[i][min])/(range[i][max]-range[i][min])
            if low > 0:
                Ps[i] = random.random()*(1.0-low)+low


def cxTwoPointCopy(ind1, ind2):
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()

    return ind1, ind2

def evaluate(ps,net,us):
    us_net = net.feedforward(np.reshape(ps,(len(ps),1)))
    r_net = 0
    for i in range(len(us)):
        r_net += np.power((us_net[i] - us[i])/us[i],2)
    k = np.sqrt(r_net/len(us))
    return k


def paraGA(length,net,us,numIndividual,numGeneration,crossOverPB, mutantPB):

    creator.create("FitnessMin",base.Fitness,weights = (-0.1,))
    creator.create("Individual",np.ndarray,fitness = creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individual",initPs,creator.Individual)
    toolbox.register("population",tools.initRepeat,list,toolbox.individual)

    toolbox.register("mate", cxTwoPointCopy)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta = 0.5, low = 0, up = 0, indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate",evaluate,net = net,us = us)

    pop = toolbox.population(n=numIndividual)
    fitness = map(toolbox.evaluate,pop)
    for ind,fit in zip(pop,fitness):
        ind.fitness.values = fit


    for g in range(numGeneration):
        print('generation {}'.format(g))
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossOverPB:
                toolbox.mate(child1, child2)
                adjustConstraint(child1)
                adjustConstraint(child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutantPB:
                toolbox.mutate(mutant)
                adjustConstraint(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitness = map(toolbox.evaluate,invalid_ind)
        for ind,fit in zip(invalid_ind,fitness):
            ind.fitness.values = fit
        # The population is entirely replaced by the offspring
        pop[:] = offspring

        bestInd = tools.selBest(pop,1)
        print(bestInd[0].fitness)
        if bestInd[0].fitness.values[0] < 0.01:
            break

    return np.reshape(bestInd[0],(len(bestInd[0]),1))

def isInRange(ps,min,max):
    for p in ps:
        if p <= min or p >= max:
            return False
    return True