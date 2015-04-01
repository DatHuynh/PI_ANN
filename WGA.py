__author__ = 'DatHuynh'
from deap import base, creator,tools
import random
import Network as nw
import DataSetGenerator as dg

def createNetwork(sizes):
    return nw.Network(sizes,0.01,0.001)

def evaluate(individual,trainingdata,numGDMStep,eta):
    net = individual[0]
    net.GD(trainingdata,None,numGDMStep,eta,isReduce=False)
    return net.evaluate(trainingdata),

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

def weightGA(sizes,trainingdata,testdata,eta ,numIndividual,numGeneration,numGDMStep,crossOverPB, mutantPB):

    print('Weight GA')

    creator.create("FitnessMin",base.Fitness,weights = (-0.1,))
    creator.create("Individual",list,fitness = creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("Init",createNetwork,sizes)
    toolbox.register("individual",tools.initRepeat,creator.Individual, toolbox.Init,1)
    toolbox.register("population",tools.initRepeat,list,toolbox.individual)

    toolbox.register("mate", cxTwoPointCopy)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate",evaluate,trainingdata = trainingdata, testdata = testdata,numGDMStep = numGDMStep)

    pop = toolbox.population(n=numIndividual)
    nW,nB = 0,0
    for i in range(len(sizes)):
        if i != len(sizes) - 1:
            nW += sizes[i]*sizes[i+1]
        if i != 0:
            nB += sizes[i]

    #CXPB, MUTPB= 0.5, 0.2

    print('Initializing')
    # Evaluate the entire population
    '''
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    '''
    for ind in pop:
        ind.fitness.values = evaluate(ind,trainingdata,numGDMStep,eta)



    if testdata is not None:
        bestInd = tools.selBest(pop,1)
        print("TrainData: {} TestData: {}".format(bestInd[0].fitness,bestInd[0][0].evaluate(testdata)))

    wlist1 = [0 for i in range(nW)]
    wlist2 = [0 for i in range(nW)]
    blist1 = [0 for i in range(nB)]
    blist2 = [0 for i in range(nB)]

    wlist = [0 for i in range(nW)]
    blist = [0 for i in range(nB)]

    for g in range(numGeneration):
        print('generation {}'.format(g))
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossOverPB:
                net1 = child1[0]
                net2 = child2[0]
                convertToArray(net1.weights,wlist1)
                convertToArray(net2.weights,wlist2)
                convertToArray(net1.biases,blist1)
                convertToArray(net2.biases,blist2)
                toolbox.mate(wlist1, wlist2)
                toolbox.mate(blist1, blist2)
                convertToWeights(wlist1,net1.weights)
                convertToWeights(wlist2,net2.weights)
                convertToWeights(blist1,net1.biases)
                convertToWeights(blist2,net2.biases)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutantPB:
                net = mutant[0]
                convertToArray(net.weights,wlist)
                convertToArray(net.biases,blist)
                toolbox.mutate(wlist)
                toolbox.mutate(blist)
                convertToWeights(wlist,net.weights)
                convertToWeights(blist,net.biases)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        for ind in invalid_ind:
            ind.fitness.values = evaluate(ind,trainingdata,numGDMStep,eta)

        # The population is entirely replaced by the offspring
        pop[:] = offspring
        bestInd = tools.selBest(pop,1)
        if testdata is not None:
            print("TrainData: {} TestData: {}".format(bestInd[0].fitness , bestInd[0][0].evaluate(testdata)))
        else:
            print("TrainData: {}".format(bestInd[0].fitness))
    return bestInd[0][0]

