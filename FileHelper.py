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

def loadUsContinue(fu,sizeU,minU,maxU):
    us = np.zeros((sizeU,1))
    counter = 0
    for i in range(0,sizeU,4):
        for e in fu.readline().split():
            us[counter] = normalize(np.float64(e),minU,maxU)
            if us[counter] < 0:
                print('Negative number Us')
            counter += 1
    return us


def loadPsContinue(fp, maxP, minP, sizeP):
    ps = np.zeros((sizeP,1))
    for i,e in enumerate(ps):
        ps[i] = normalize(np.float64(fp.readline()), minP, maxP)
        if ps[i] < 0:
            print('Negative number Ps')
    return ps


def loadTrainingReal(filenameP,filenameU, sizeP,sizeU):
    fp = open(filenameP+'.txt','r')
    fu = open(filenameU+'.txt','r')
    dataset = []
    while fp.readline() is not '' and fu.readline() is not '':
        data = []
        ps = loadPsContinue(fp, 0, 0, sizeP)
        us = loadUsContinue(fu,sizeU,0,0)
        data.append(np.reshape(ps,(sizeP,1)))
        data.append(np.reshape(us,(sizeU,1)))
        dataset.append(data)
    fp.close()
    fu.close()
    return dataset

def loadUs(filename,size,minU,maxU):
    f = open(filename+'.txt','r')
    f.readline()
    us = loadUsContinue(f,size,minU,maxU)
    f.close()
    return us

def loadPs(filename,size):
    f = open(filename+'.txt','r')
    f.readline()
    ps = loadPsContinue(f,0,0,size)
    f.close()
    return ps

def saveVecContinue(f,vec,idx):
    f.write('-----Data{}\n'.format(idx))
    for u in vec:
        f.write(str(u[0]))
        f.write('\n')

def saveVec(filename,vec):
    f = open(filename+'.txt','w')
    saveVecContinue(f,vec,0)
    f.close()

def saveTrainingReal(filenameP,filenameU,dataset):
    fp = open(filenameP+'.txt','w')
    fu = open(filenameU+'.txt','w')
    for i,data in enumerate(dataset):
        saveVecContinue(fp,data[0],i)
        saveVecContinue(fu,data[1],i)
    fp.close()
    fu.close()

def normalize(e,min,max):
    return e
    #return (e-min)/(max-min)


def deNormalize(e,min,max):
    return e
    #return e*(max-min)+min

def FormatPs(filenameO,filenameW,dimData,numData):

    f = open(filenameO+'.txt','r')
    arr = []
    for i in range(dimData):
        arr.append(f.readline().split())
    f.close()

    #sanity check
    if len(arr) != dimData:
        print('Insane')
    if len(arr[0]) != numData:
        print('Insane 2')

    f = open(filenameW+'.txt','w')
    for i in range(numData):
        f.write('-----Data'+str(i)+'\n')
        for j in range(dimData):
            f.write(arr[j][i]+'\n')
    f.close()

def FormatUs(filenameO,filenameW,width,height,numData):

    f = open(filenameO+'.txt','r')
    arr = []
    for i in range(height):
        arr.append(f.readline().split())
    f.close()

    #sanity check
    if len(arr) != height:
        print('Insane')
    if len(arr[0]) != width*numData:
        print('Insane 2')

    f = open(filenameW+'.txt','w')
    for i in range(numData):
        f.write('-----Data'+str(i)+'\n')
        for j in range(height):
            for k in range(width):
                f.write(arr[j][i*width+k]+' ')
            f.write('\n')
    f.close()


#Test
#FormatPs('RawPs','PsBeta',10,29)
#FormatUs('RawUs','UsBeta',4,35,29)