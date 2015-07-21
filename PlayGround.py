__author__ = 'DatHuynh'
import DataSetGenerator as dg
import WGA
import Network as nw
import PGD
import FileHelper as fh
import PGA
import numpy as np

ps = None
sizes = [10,8,140]
trainingdata = fh.loadTrainingReal('PsBeta','UsBeta',sizes[0],sizes[2])            #this is the slow down factor
testdata = None
usTarget = fh.loadUs('UsTarget',sizes[2],0.1,31)
success = False
t = 1
thresholdTerminate = 0.1
eta = 0.03
#while success is False:
#800
print(len(trainingdata))

print("#PI {}".format(t))
net = WGA.weightGA(sizes,trainingdata,testdata,eta=eta,numIndividual= 50,numGeneration=800,numGDMStep=10,crossOverPB= 0.5, mutantPB= 0.5)
fh.saveWeightJson(net.weights,"WeightsRaw")
fh.saveWeightJson(net.biases,"BiasesRaw")


#threshold is better at 0.028
net = nw.Network(sizes,threshold= thresholdTerminate,alpha=0.1)
net.weights = fh.loadWeightJson("WeightsRaw")
net.biases = fh.loadWeightJson("BiasesRaw")

print("End Of WGA Pharse - TrainData: {}".format(net.evaluate(trainingdata)))

#100000
net.GD(trainingdata,None,epoch=10000,eta=eta)
print("Save weight")
fh.saveWeightJson(net.weights,"WeightsOpti")
fh.saveWeightJson(net.biases,"BiasesOpti")

net.GD(trainingdata,None,epoch=10000,eta=eta)
print("Save weight")
fh.saveWeightJson(net.weights,"WeightsOpti")
fh.saveWeightJson(net.biases,"BiasesOpti")

net.GD(trainingdata,None,epoch=20000,eta=eta)
print("Save weight")
fh.saveWeightJson(net.weights,"WeightsOpti")
fh.saveWeightJson(net.biases,"BiasesOpti")

net = nw.Network(sizes,threshold= thresholdTerminate,alpha=0.001)
net.weights = fh.loadWeightJson("WeightsOpti")
net.biases = fh.loadWeightJson("BiasesOpti")
print("End Of W optimize Pharse - TrainData: {}".format(net.evaluate(trainingdata)))
#100

ps = PGA.paraGA(sizes[0],net,usTarget, numIndividual=500, numGeneration=500,crossOverPB=0.5,mutantPB=0.3)
fh.saveVec('PsSuggest',ps)
print("End Of PGA phrase - NetError: {}".format(PGA.evaluate(ps,net,usTarget)))
#10000

ps = fh.loadPs('PsSuggest',sizes[0])
ps = PGD.optimizePs(ps,net,usTarget,epoch=10000,eta=0.003)       #eta is not good -- dynamic eta
fh.saveVec('PsSuggestOpti',ps)
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
'''

net = nw.Network(sizes,threshold= thresholdTerminate,alpha=0.001)
net.weights = fh.loadWeightJson("WeightsOpti")
net.biases = fh.loadWeightJson("BiasesOpti")
print("End Of W optimize Pharse - TrainData: {}".format(net.evaluate(trainingdata)))

ps = fh.loadPs('PsSuggestOpti',sizes[0])
print("End Of P optimize phrase - NetError: {}".format(PGA.evaluate(ps,net,usTarget)))
'''