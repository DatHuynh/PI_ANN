__author__ = 'DatHuynh'
import DataSetGenerator as dg
import WGA
import Network as nw
import PGD
import FileHelper as fh
import PGA
import numpy as np

'''
Haven't normalize
'''
ps = None
sizes = [20,10,102]
#answerPs = [0.2664499,0.5221133,0.88982]
#10,216000,1.12,30.31
trainingdata = fh.loadTrainingReal('Ps','Us',sizes[0],sizes[2],9,215999,0.1,31)            #this is the slow down factor
testdata = None
usTarget = fh.loadUs('UsTarget',sizes[2],0.1,31)
success = False
t = 1

#while success is False:
#800
'''
print("#PI {}".format(t))
net = WGA.weightGA(sizes,trainingdata,testdata,eta=0.001,numIndividual= 50,numGeneration=800,numGDMStep=10,crossOverPB= 0.5, mutantPB= 0.2)
fh.saveWeightJson(net.weights,"WeightsRaw")
fh.saveWeightJson(net.biases,"BiasesRaw")

print("End Of WGA Pharse - TrainData: {}".format(net.evaluate(trainingdata)))


#threshold is better at 0.028
net = nw.Network(sizes,threshold= 0.02,alpha=0.001)
net.weights = fh.loadWeightJson("WeightsRaw")
net.biases = fh.loadWeightJson("BiasesRaw")

#100000
net.GD(trainingdata,None,epoch=100000,eta=0.001)
fh.saveWeightJson(net.weights,"WeightsOpti")
fh.saveWeightJson(net.biases,"BiasesOpti")

'''
net = nw.Network(sizes,threshold= 0.02,alpha=0.001)
net.weights = fh.loadWeightJson("WeightsOpti")
net.biases = fh.loadWeightJson("BiasesOpti")
#print("End Of W optimize Pharse - TrainData: {}".format(net.evaluate(trainingdata)))
#100
#ps = PGA.paraGA(sizes[0],net,usTarget, numIndividual=100, numGeneration=500,crossOverPB=0.5,mutantPB=0.3)
#fh.saveVec('PsSuggest',ps)
#print("End Of PGA phrase - NetError: {}".format(PGA.evaluate(ps,net,usTarget)))
#10000
ps = fh.loadPs('PsSuggest',sizes[0])
#ps = PGD.optimizePs(ps,net,usTarget,epoch=10000,eta=0.001)       #eta is not good -- dynamic eta
#fh.saveVec('PsSuggestOpti',ps)
print("End Of P optimize phrase - NetError: {}".format(PGA.evaluate(ps,net,usTarget)))

confirm = input('Please enter Us Model: ')

if confirm == 'Yes':
    usModel = fh.loadUs('UsModel',sizes[2],1,31)

    tolerance = PGD.evaluateTolerance(ps,net,usModel,usTarget)

    print('Tolerance: {}'.format(tolerance))

    if(tolerance < 0.01):
        print('Success')

    fh.saveTrainingReal('Ps{}'.format(len(trainingdata)+1),'Us{}'.format(len(trainingdata)+1),trainingdata)
    #net.feedforward(np.reshape([0.1,0.1,0.3],(3,1)))