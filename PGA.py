__author__ = 'DatHuynh'

from deap import base, creator,tools
import random
import numpy as np
import Network as nw
import DataSetGenerator as dg

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
    r_dg,r_net = 0,0
    for i in range(len(us)):
        r_net += np.power((us_net[i] - us[i])/us[i],2)
    return np.sqrt(r_net/len(us))


def paraGA(length,net,us):

    creator.create("FitnessMin",base.Fitness,weights = (-0.1,))
    creator.create("Individual",np.ndarray,fitness = creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individual",tools.initRepeat,creator.Individual, random.random,length)
    toolbox.register("population",tools.initRepeat,list,toolbox.individual)

    toolbox.register("mate", cxTwoPointCopy)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate",evaluate,net = net,us = us)

    pop = toolbox.population(n=1000)
    fitness = map(toolbox.evaluate,pop)
    for ind,fit in zip(pop,fitness):
        ind.fitness.values = fit

    CXPB, MUTPB, NGEN = 0.5, 0.2, 50

    for g in range(NGEN):
        print('generation {}'.format(g))
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
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

    return np.reshape(bestInd[0],(len(bestInd[0]),1))