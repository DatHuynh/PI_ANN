import FileHelper as fh
import numpy as np
import Network as network
import PGD
import DataSetGenerator as dg

'''
ps = np.reshape([ 0.14093579,0.63546226, 0.06332656],(3,1))
us = np.reshape(dg.computeUs(0.2664499,0.5221133,0.88982),(1,1))

net = network.Network([3,3,1],0.02,0.001)
net.weights = fh.loadWeightJson('WeightsOpti')
net.biases = fh.loadWeightJson('BiasesOpti')

print(PGD.optimizePs(ps,net,us,epoch=10000,eta=0.1))
'''

trainingdata = fh.loadTrainingReal('Ps','Us',20,204,9,215999,1,31)            #this is the slow down factor
