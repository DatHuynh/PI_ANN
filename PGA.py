__author__ = 'DatHuynh'

from deap import base, creator,tools
import random
import numpy as np
import Network as nw
import DataSetGenerator as dg

sizes = [3,4,1]

def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwritting in the swap operation. It prevents
    ::

        >>> import numpy
        >>> a = numpy.array((1,2,3,4))
        >>> b = numpy.array((5.6.7.8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
    """
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
    us_net = net.feedforward(ps)
    r_dg,r_net = 0,0
    for i in range(us):
        r_net = np.power((us_net[i] - us[i])/us[i],2)
    return np.sqrt(r_net/len(us))

creator.create("FitnessMin",base.Fitness,weights = (-0.1,))
creator.create("Individual",np.ndarray,fitness = creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("individual",tools.initRepeat,creator.Individual, random.random,sizes[0])
toolbox.register("population",tools.initRepeat,list,toolbox.individual)

toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

def paraGA(net,us):
    pop = toolbox.population(n=50)
    for ind in pop:
        ind.fitness.values = evaluate(ind,net,us)

    CXPB, MUTPB, NGEN = 0.5, 0.2, 100

    for g in range(NGEN):


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

        print('generation {}'.format(g))
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = evaluate(ind,net,us)

        bestInd = tools.selBest(pop,1)
        print(bestInd[0].fitness)

        # The population is entirely replaced by the offspring
        pop[:] = offspring
    return bestInd[0]