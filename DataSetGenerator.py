__author__ = 'DatHuynh'
import numpy as np
import random

def computeMeasure1(kdp, kdi, xD, xd):
    if (kdi + (kdp - kdi) * np.sqrt(xD * (2 * xd - xD + 2) / (xd * xd))) < 0:
        print("Fail nonnegative")
    return (kdi + (kdp - kdi) * np.sqrt(xD * (2 * xd - xD + 2) / (xd * xd)))/20


def computeMeasure2(kdp, kdr, xD, xd):
    if ((kdp - kdr) * np.exp(-np.power((xD - xd) / xd, 2)) + kdr)/20 < 0:
        print("Fail nonnegative")
    return ((kdp - kdr) * np.exp(-np.power((xD - xd) / xd, 2)) + kdr)/20


def computeUs(p1, p2, p3):
    #u = np.sin(p1 * 3.14 / 2) * np.exp(p2)/2 + np.sqrt(p3 * 30)/11
    u = np.sin(p1 * 3.14 / 2) * np.exp(p2)/((p3+1)*30)
    return u


def generateData(p1,p2,p3,p4):
    kdi = computeK_prime(p1)
    kdp = computeK_prime(p2)
    kdr = computeK_prime(p3)
    xd = computeX(p2,p3)
    xD = p4

    factors = [6,7,8,9]
    ds = [0.1,1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1]
    ps = [p1,p2,p3]
    us = []
    data = []
    '''
    for f in factors:
        for d in ds:
            if xD > xd:
                u = f*computeMeasure1(kdi*100, kdp, xD, xd+d)/1000
            else:
                u = f*computeMeasure2(kdp*100, kdr, xD, xd+d)/1000
            if u > 1 or u < 0:
                print("Fail at Normalize")
            us.append(u)
    '''
    u = computeUs(p1, p2, p3)
    us.append(u)
    if u > 1 or u < 0:
        print("Fail at Normalize p1: {} p2: {} p3: {} u: {}".format(p1,p2,p3,u))
    data.append(np.reshape(ps,(len(ps),1)))
    data.append(np.reshape(us,(len(us),1)))
    return data

#normalize
def computeK_prime(p):
    return (2*np.sin(p))/(3+np.sin(p))*3/2
#normalize
def computeX(p1,p2):
    return 1/((computeK_prime(p1*3.14/2)-computeK_prime(p2*3.14/2)+1)*69)

def generateTrainingData():
    dataset = []
    p1s = [0.1,0.9]
    p2s = [0.1,0.9]
    p3s = [0.1,0.9]
    for p1 in p1s:
        for p2 in p2s:
            for p3 in p3s:
                #for p4 in range (1,11,1):
                dataset.append(generateData(p1,p2,p3,0.35))

    #scatter data
    '''
    dataset.append(generateData(0.33,0.44,0.55,0.35))       #fail :)
    dataset.append(generateData(0.12,0.35,0.69,0.35))
    dataset.append(generateData(0.08,0.24,0.66,0.35))
    dataset.append(generateData(0.15,0.11,0.36,0.35))
    dataset.append(generateData(random.random(),random.random(),random.random(),random.random()))
    dataset.append(generateData(random.random(),random.random(),random.random(),random.random()))
    dataset.append(generateData(random.random(),random.random(),random.random(),random.random()))
    dataset.append(generateData(random.random(),random.random(),random.random(),random.random()))
    '''
    return dataset


def generateTestData():
    dataset = []
    for p1 in range(1,10):
        for p2 in range(1,10):
            for p3 in range(1,10):
                #for p4 in range (1,11,1):
                dataset.append(generateData(p1*0.1,p2*0.1,p3*0.1,0.35))
    return dataset
'''
def generateTestData():
    dataset = []
    for p1 in range(1,12,1):
        for p2 in range(3,25,2):
            for p3 in range(9,42,3):
                #for p4 in range (1,11,1):
                dataset.append(generateData(0.1*p1,0.1*p2,0.1*p3,0.35))
    return dataset

def generateTrainingData():
    dataset = []
    p1s = [0.1,1.1]
    p2s = [0.3,2.3]
    p3s = [0.9,3.9]
    for p1 in p1s:
        for p2 in p2s:
            for p3 in p3s:
                #for p4 in range (1,11,1):
                dataset.append(generateData(p1,p2,p3,0.35))
    return dataset
'''