__author__ = 'DatHuynh'
import DataSetGenerator as dg
import WGA
import PGD
import FileHelper as fh
import PGA
import numpy as np

sizes = [3,4,1]
answerPs = [0.35,0.69,0.88]
us = np.reshape(0.039440896645075559,(1,1))
trainingdata = dg.generateTrainingData()
testdata = dg.generateTestData()

'''
net1 =  network1.Network([3,3,40])
net1.SGD(trainingdata,100000,8,1,testdata)
print(net1.evaluate(testdata))
'''

net2 = WGA.weightGA()
fh.saveWeightJson(net2.weights,"WeightsRaw")
fh.saveWeightJson(net2.biases,"BiasesRaw")

net2.GD(trainingdata,testdata[:100],100,True)
fh.saveWeightJson(net2.weights,"WeightsOpti")
fh.saveWeightJson(net2.biases,"BiasesOpti")

ps = PGA.paraGA(net2,us)

PGD.optimizePs(ps,net2,us,1000)
#net2.feedforward(np.reshape([0.1,0.1,0.3],(3,1)))