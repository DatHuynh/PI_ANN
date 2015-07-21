__author__ = 'DatHuynh'
import DataSetGenerator as dg
import WGA
import Network as nw
import PGD
import FileHelper as fh
import PGA
import numpy as np
import math

def foldSplit(arr,numFold, FoldId):
    if FoldId >= numFold:
        return None
    numEle = math.floor(len(arr)/numFold)
    if FoldId == numFold - 1:
        train_data = arr[0:FoldId*numEle]
        validate_data = arr[FoldId*numEle:]
    else:
        validate_data = arr[FoldId*numEle:(FoldId+1)*numEle]
        train_data = arr[0:FoldId*numEle]
        train_data += arr[(FoldId+1)*numEle:]
    return train_data,validate_data

sizes = [10,8,140]
answerPs = [0.33,0.44,0.55]
trainingdata = fh.loadTrainingReal('PsBeta','UsBeta',sizes[0],sizes[2],9,215999,0.1,31);
lds = [0.000003,0.00001,0.00003,0.0001,0.0003,0.001]
etas = [0.000003,0.00001,0.00003,0.0001,0.0003,0.001]
hiddenN = [8,10,12,14,16,18,20,22,24,26,28]
result = {}
best_ld = 0
best_eta = 0
best_hidden = 0
min_loss = 10   #only take the para with loss smaller than 10
numFold = 5

'''
for ld in lds:
    for eta in etas:
        loss = 0
        for id in range(numFold):-
            print('ld:{} eta:{} foldID:{}'.format(ld,eta,id))
            foldTrain, foldValidate = foldSplit(trainingdata,numFold,id)
            net = nw.Network(sizes,threshold= 0.02,alpha=0.001,ld = ld,eta = eta)
            net.GD(foldTrain,None,epoch=100,isReduce=False)
            loss += net.evaluate(foldValidate)/numFold
        result[(ld,eta)] = loss
        if loss < min_loss:
            min_loss = loss
            best_eta = eta
            best_ld = ld
'''

for hidden in hiddenN:
    sizes[1] = hidden
    loss = 0
    for id in range(numFold):
        print("size "+ str(sizes))
        foldTrain, foldValidate = foldSplit(trainingdata,numFold,id)
        '''
        net = nw.Network(sizes,threshold= 0.02,alpha=0.001)
        net.GD(foldTrain,None,epoch=1000,isReduce=False,eta=0.05)
        '''
        net = nw.Network(sizes,threshold= 0.02,alpha=0.001)
        net.GD(foldTrain,None,1000,eta = 0.05, lmbda=0.1)

        loss += net.evaluate(foldValidate)/numFold
    result[hidden] = loss
    if loss < min_loss:
        min_loss = loss
        best_hidden = hidden

print(result)

print('best hidden{}:'.format(best_hidden))
sizes[1] = best_hidden
net = nw.Network(sizes,threshold= 0.02,alpha=0.001)
net.GD(trainingdata,None,4000,eta=0.03,lmbda=0)

print(net.evaluate(trainingdata))