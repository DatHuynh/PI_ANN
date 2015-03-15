__author__ = 'DatHuynh'
import DataSetGenerator as dg
import SampleNetwork as network1
import Network as network2
import GA
import PGD
trainingdata = dg.generateTrainingData()
testdata = dg.generateTestData()

'''
net1 =  network1.Network([3,3,40])
net1.SGD(trainingdata,100000,8,1,testdata)
print(net1.evaluate(testdata))
'''

net2 = GA.weightGA()
net2.GD(trainingdata,testdata[:100],100000,True)

PGD.optimizePs([0.1,0.1,0.1],net2,[0.09],1000)
#net2.feedforward(np.reshape([0.1,0.1,0.3],(3,1)))