import FileHelper as fh
import numpy as np
import Network as network
import PGD
import DataSetGenerator as dg

ps = np.reshape([0.35917884261010347, 0.5683671133654195,0.864315400434641],(3,1))
us = np.reshape(dg.computeUs(0.33,0.44,0.55),(1,1))

net = network.Network([3,3,1],0.02,0.001)
net.weights = fh.loadWeightJson('WeightsOpti')
net.biases = fh.loadWeightJson('BiasesOpti')

PGD.optimizePs(ps,net,us,epoch=10000,eta=0.001)

print(PGD.evaluateTolerance(ps,net,us))