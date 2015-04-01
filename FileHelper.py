__author__ = 'DatHuynh'
import json
import numpy as np
import DataSetGenerator as dg

def arr2json(arr):
    return json.dumps(arr.tolist())


def json2arr(data):
    return np.array(json.loads(data),dtype = np.float64)


def saveWeightJson(weights ,filename):
    f = open(filename+'.txt','w')
    for mat in weights:
        f.write(arr2json(mat)+'\n')
    f.close()


def loadWeightJson(filename):
    f = open(filename+'.txt','r')
    data = f.read()
    f.close()
    eles = list(filter(None,data.split('\n')))
    weights = []
    for ele in eles:
        weights.append(json2arr(ele))
    return weights

def saveTrainingDataSet(trainingdata,filename):
    f = open(filename+'.txt','w')
    for data in trainingdata:
        f.write(arr2json(data[0])+'\n')
    f.close()

def loadTrainingDataSet(filename):
    trainingdata = []
    f = open(filename+'.txt','r')
    data = f.read()
    f.close()
    eles = list(filter(None,data.split('\n')))
    weights = []
    for ele in eles:
        trainingdata.append(dg.generateData(ele[0],ele[1],ele[2],0.35))
    return trainingdata

def loadUs(fu,sizeU):
    us = np.zeros((sizeU,1))
    counter = 0
    for i in range(0,sizeU,3):
        for e in fu.readline().split():
            us[counter] = np.float64(e)
            counter += 1
    return us


def loadTrainingReal(filenameP,filenameU, sizeP,sizeU):
    fp = open(filenameP+'.txt','r')
    fu = open(filenameU+'.txt','r')
    dataset = []
    while fp.readline() is not '' and fu.readline() is not '':
        data = []
        ps = [np.float64(fp.readline()) for i in range(sizeP)]
        us = loadUs(fu,sizeU)
        data.append(np.reshape(ps,(sizeP,1)))
        data.append(np.reshape(us,(sizeU,1)))
        dataset.append(data)
    fp.close()
    fu.close()
    return dataset

def loadTargetUs(filename,size):
    f = open(filename+'.txt','r')
    us = loadUs(f,size)
    f.close()
    return us