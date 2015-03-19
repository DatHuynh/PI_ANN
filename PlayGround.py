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
trainingdata = dg.generateTrainingData()
testdata = dg.generateTestData()

net = WGA.weightGA(sizes,trainingdata,testdata[100:200],50,1250,10,0.5,0.3)
fh.saveWeightJson(net.weights,"WeightsRaw")
fh.saveWeightJson(net.biases,"BiasesRaw")

net = nw.Network(sizes,0.01,0.001)
net.weights = fh.loadWeightJson("WeightsRaw")
net.biases = fh.loadWeightJson("BiasesRaw")

net.GD(trainingdata,testdata[:100],1000,True)
#fh.saveWeightJson(net.weights,"WeightsOpti")
#fh.saveWeightJson(net.biases,"BiasesOpti")

ps = PGA.paraGA(net,us)

PGD.optimizePs(ps,net,us,100000)
#net.feedforward(np.reshape([0.1,0.1,0.3],(3,1)))