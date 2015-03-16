__author__ = 'DatHuynh'
from deap import base, creator,tools
import random
import Network as nw
import DataSetGenerator as dg

sizes = [3,4,1]

def createNetwork():
    return nw.Network(sizes,0.01,0.001)

def evaluate(individual):
    net = individual[0]
    net.GD(trainingdata,testdata[:100],10)
    return net.evaluate(testdata),

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

creator.create("FitnessMin",base.Fitness,weights = (-0.1,))
creator.create("Individual",list,fitness = creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("individual",tools.initRepeat,creator.Individual, createNetwork,1)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)

toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

trainingdata = dg.generateTrainingData()
testdata = dg.generateTestData()

def convertToWeights(array,weights):
    m = 0
    for i,weight in enumerate(weights):
        for k,e in enumerate(weight):
            weights[i][k] = array[m:m+len(e)]
            m+= len(e)

def convertToArray(weights,array):
    m = 0
    for i,weight in enumerate(weights):
        for k,e in enumerate(weight):
            array[m:m+len(e)] = weights[i][k]
            m+= len(e)

def weightGA():
    pop = toolbox.population(n=50)
    networks = []
    nW,nB = 0,0
    for i in range(len(sizes)):
        if i != len(sizes) - 1:
            nW = sizes[i]*sizes[i+1]
        nB += sizes[i]

    CXPB, MUTPB, NGEN = 0.5, 0.2, 1250

    print('Initializing')
    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    bestInd = tools.selBest(pop,1)
    print(bestInd[0].fitness)

    for g in range(NGEN):


        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                wlist1 = [0 for i in range(nW)]
                wlist2 = [0 for i in range(nW)]
                blist1 = [0 for i in range(nB)]
                blist2 = [0 for i in range(nB)]
                net1 = child1[0]
                net2 = child2[0]
                convertToArray(net1.weights,wlist1)
                convertToArray(net2.weights,wlist2)
                convertToArray(net1.weights,blist1)
                convertToArray(net2.weights,blist2)
                toolbox.mate(wlist1, wlist2)
                toolbox.mate(blist1, blist2)
                convertToWeights(wlist1,net1.weights)
                convertToWeights(wlist2,net2.weights)
                convertToWeights(blist1,net1.biases)
                convertToWeights(blist2,net2.biases)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                wlist = [0 for i in range(nW)]
                blist = [0 for i in range(nB)]
                net = mutant[0]
                convertToArray(net.weights,wlist)
                convertToArray(net.biases,blist)
                toolbox.mutate(wlist)
                toolbox.mutate(blist)
                convertToWeights(wlist,net.weights)
                convertToWeights(blist,net.biases)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness

        print('generation {}'.format(g))
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        bestInd = tools.selBest(pop,1)
        print(bestInd[0].fitness)

        # The population is entirely replaced by the offspring
        pop[:] = offspring
    return bestInd[0][0]

