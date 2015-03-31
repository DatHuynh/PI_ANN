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