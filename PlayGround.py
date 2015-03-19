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

'''
net1 =  network1.Network([3,3,40])
net1.SGD(trainingdata,100000,8,1,testdata)
print(net1.evaluate(testdata))
'''
'''
net2 = WGA.weightGA()
fh.saveWeightJson(net2.weights,"WeightsRaw")
fh.saveWeightJson(net2.biases,"BiasesRaw")
'''
net2 = nw.Network(sizes,0.01,0.001)
net2.weights = fh.loadWeightJson("WeightsRaw")
net2.biases = fh.loadWeightJson("BiasesRaw")
'''
#net2.GD(trainingdata,testdata[:100],1000,True)
#fh.saveWeightJson(net2.weights,"WeightsOpti")
#fh.saveWeightJson(net2.biases,"BiasesOpti")
'''
ps = PGA.paraGA(net2,us)

PGD.optimizePs(ps,net2,us,100000)
#net2.feedforward(np.reshape([0.1,0.1,0.3],(3,1)))