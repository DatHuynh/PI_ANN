__author__ = 'DatHuynh'
import numpy as np
import Network as nw
import DataSetGenerator as dg


'''
def optimizePs(ps,net,us,epoch,eta):
    psPre = np.zeros((1,1))
    preTolerance, curTolerance = 0,0
    for i in range(epoch):
        delta = PGD(ps,net,us)
        psPre = ps = updatePs(ps,delta,eta)
        print(evaluateTolerance(ps,net,us))
        #print(evaluate(ps,net,us))
        print(ps)
        if(not isInRange(ps,0,1)):
            return psPre
    return ps

'''

def optimizePs(ps,net,us,epoch,eta):
    psPre = None
    preError = -1
    for i in range(epoch):
        delta = PGD(ps,net,us)
        ps = updatePs(ps,delta,eta)
        curError = evaluate(ps,net,us)
        print('Ps: {} Error: {} Tolerance: {}'.format(ps,curError,evaluateTolerance(ps,net,us)))
        if preError != -1 and curError > preError:
            ps = psPre
            eta /= 2
            print('Adjust eta')
        if(not isInRange(ps,0,1)):
            return psPre
        psPre = ps
        preError = curError
    return ps

def evaluateTolerance(ps,net,us):
    us_dg = dg.computeUs(ps[0],ps[1],ps[2])
    us_net = net.feedforward(np.reshape(ps,(len(ps),1)))
    r_dg,r_net = 0,0
    for i in range(len(us)):
        r_dg += np.power((us_dg[i] - us[i])/us[i],2)
        r_net += np.power((us_net[i] - us[i])/us[i],2)
    tolerance = np.abs(np.sqrt(r_dg/len(us)) -np.sqrt(r_net/len(us)))
    return tolerance

def evaluate(ps,net,us):
    us_net = net.feedforward(np.reshape(ps,(len(ps),1)))
    r_dg,r_net = 0,0
    for i in range(len(us)):
        r_net += np.power((us_net[i] - us[i])/us[i],2)
    return np.sqrt(r_net/len(us))

def PGD(ps, net, us):
    activation = ps
    activations = [ps]
    zs = []
    for b,w in zip(net.biases,net.weights):
        z = np.dot(w,activation)+b
        zs.append(z)
        activation = sigmoid_vec(z)
        activations.append(activation)

    delta = cost_derivative(activations[-1],us)*sigmoid_prime_vec(zs[-1])

    for l in range(2,net.numLayers):
        delta = np.dot(net.weights[-l+1].transpose(),delta)*sigmoid_prime_vec(zs[-l])

    delta = np.dot(net.weights[0].transpose(),delta)
    return delta

def updatePs(ps,delta,eta):
    ps = [p - eta*deltap for p,deltap in zip(ps,delta)]
    return ps
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid)

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

sigmoid_prime_vec = np.vectorize(sigmoid_prime)

def cost_derivative(activation,y):
        return 2*(activation-y)/(y*y)
        #return 2*(activation-y)

def isInRange(ps,min,max):
    for p in ps:
        if p <= min or p >= max:
            return False
    return True