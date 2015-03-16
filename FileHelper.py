__author__ = 'DatHuynh'
import json
import numpy as np

def arr2json(arr):
    return json.dumps(arr.tolist())


def json2arr(data):
    return np.array(json.loads(data))


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