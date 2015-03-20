__author__ = 'DatHuynh'
import DataSetGenerator as dg
import WGA
import Network as nw
import PGD
import FileHelper as fh
import PGA
import numpy as np

sizes = [3,4,1]
answerPs = [0.33,0.44,0.55]
us = np.reshape(dg.computeUs(answerPs[0],answerPs[1],answerPs[2]),(1,1))
trainingdata = dg.generateTrainingData()            #this is the slow down factor
testdata = dg.generateTestData()
'''
net = WGA.weightGA(sizes,trainingdata,testdata,eta=0.01,numIndividual= 50,numGeneration=1250,numGDMStep=10,crossOverPB= 0.5, mutantPB= 0.2)
fh.saveWeightJson(net.weights,"WeightsRaw")
fh.saveWeightJson(net.biases,"BiasesRaw")
'''

#threshold is better at 0.028
net = nw.Network(sizes,threshold= 0.03,alpha=0.001)
net.weights = fh.loadWeightJson("WeightsRaw")
net.biases = fh.loadWeightJson("BiasesRaw")

print(net.evaluate(testdata))

net.GD(trainingdata,testdata,epoch=100000,eta=0.1,isReport=True)
fh.saveWeightJson(net.weights,"WeightsOpti")
fh.saveWeightJson(net.biases,"BiasesOpti")

net.weights = fh.loadWeightJson("WeightsOpti")
net.biases = fh.loadWeightJson("BiasesOpti")
ps = PGA.paraGA(sizes[0],net,us)

PGD.optimizePs(ps,net,us,100000,0.01)
#net.feedforward(np.reshape([0.1,0.1,0.3],(3,1)))