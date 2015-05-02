__author__ = 'DatHuynh'
import DataSetGenerator as dg
import WGA
import Network as nw
import PGD
import FileHelper as fh
import PGA
import numpy as np

sizes = [3,3,1]
answerPs = [0.33,0.44,0.55]
trainingdata = dg.generateTrainingData()            #this is the slow down factor
testdata = dg.generateTestData()
us = np.reshape(dg.computeUs(answerPs[0],answerPs[1],answerPs[2]),(1,1))
success = False
'''
while success is False:
    net = WGA.weightGA(sizes,trainingdata,testdata,eta=0.001,numIndividual= 50,numGeneration=500,numGDMStep=10,crossOverPB= 0.5, mutantPB= 0.2)
    fh.saveWeightJson(net.weights,"WeightsRaw")
    fh.saveWeightJson(net.biases,"BiasesRaw")

    print("End Of WGA Pharse - TrainData: {} TestData: {}".format(net.evaluate(trainingdata) , net.evaluate(testdata)))

    #threshold is better at 0.028
    net = nw.Network(sizes,threshold= 0.02,alpha=0.001)
    net.weights = fh.loadWeightJson("WeightsRaw")
    net.biases = fh.loadWeightJson("BiasesRaw")

    net.GD(trainingdata,None,epoch=100000,eta=0.01)
    fh.saveWeightJson(net.weights,"WeightsOpti")
    fh.saveWeightJson(net.biases,"BiasesOpti")

    print("End Of W optimize Pharse - TrainData: {} TestData: {}".format(net.evaluate(trainingdata) , net.evaluate(testdata)))

    net = nw.Network(sizes,threshold= 0.02,alpha=0.001)
    net.weights = fh.loadWeightJson("WeightsOpti")
    net.biases = fh.loadWeightJson("BiasesOpti")

    ps = PGA.paraGA(sizes[0],net,us, numIndividual=10000, numGeneration=100,crossOverPB=0.5,mutantPB=0.3)

    print("End Of PGA phrase - NetError: {} Tolerance: {}".format(PGA.evaluate(ps,net,us) ,PGD.evaluateTolerance(ps,net,us)))

    ps = PGD.optimizePs(ps,net,us,epoch=10000,eta=0.001)       #due to gradient vanishing eta could equal 1 <- wrong

    print("End Of P optimize phrase - NetError: {} Tolerance: {}".format(PGA.evaluate(ps,net,us) ,PGD.evaluateTolerance(ps,net,us)))

    if(PGD.evaluateTolerance(ps,net,us) < 0.01):
        success = True
    else:
        trainingdata.append( dg.generateData(ps[0],ps[1],ps[2],0.35))
        fh.saveTrainingDataSet(trainingdata,'trainingdata')
    #net.feedforward(np.reshape([0.1,0.1,0.3],(3,1)))
'''

