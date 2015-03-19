__author__ = 'DatHuynh'

import numpy as np

#bias crossover missing

class Network:
    def __init__(self, sizes, threshold, alpha):
        self.numLayers = len(sizes)
        self.sizes = sizes
        self.threshold = threshold
        self.alpha = alpha
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
        #self.weights = [np.zeros((y,x)) for x,y in zip(sizes[:-1],sizes[1:])]
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]

    def updateNetwork(self,training_data):
        error_w = [np.zeros(w.shape) for w in self.weights]
        error_b = [np.zeros(b.shape) for b in self.biases]
        for dataset in training_data:
            error_t_w, error_t_b = self.backProp(dataset)
            error_b = [b + deltab for b, deltab in zip(error_b, error_t_b)]
            error_w = [w + deltaw for w, deltaw in zip(error_w, error_t_w)]
        self.weights = [w - 0.01*deltaw / len(training_data) for w, deltaw in zip(self.weights, error_w)]
        self.biases = [b - 0.01*deltab / len(training_data) for b, deltab in zip(self.biases, error_b)]

    def GD(self, training_data, test_data, epoch, isReport = False):
        for i in range(epoch):
            self.updateNetwork(training_data)
            if isReport is True:
                test_error = self.evaluate(test_data)
                train_error = self.evaluate(training_data)
                print("Let see test: {} train: {}".format(test_error,train_error))
                if test_error < np.power(self.threshold,2):
                    print('Debug')
            if self.isTerminate(training_data):
                return 1
        return 0

    def reduceTraining(self, dataset, activation, delta):
        ep = (activation - dataset[1])/(dataset[1])

        for i in range(len(ep)):
            if(np.abs(ep[i]) < self.threshold):
                p = np.exp( np.log(self.alpha)*np.square(10/9)*np.square( (ep[i]-self.threshold)/self.threshold ) )
                delta[i] *= p

    def isTerminate(self,training_data):
        for dataset in training_data:
            activation = self.feedforward(dataset[0])
            ep = (activation - dataset[1])/(dataset[1])
            for i in range(len(ep)):
                if(np.abs(ep[i]) > self.threshold):
                    return False
        return True

    def feedforward(self,activation):
        for w,b in zip(self.weights,self.biases):
            z = np.dot(w,activation)+b
            activation = sigmoid_vec(z)
        return activation

    def backProp(self, dataset):
        x = dataset[0]
        y = dataset[1]

        error_w = [np.zeros(w.shape) for w in self.weights]
        error_b = [np.zeros(b.shape) for b in self.biases ]

        #feed forward
        activation = x
        activations = [x]
        zs = []
        for b,w in zip(self.biases,self.weights):
            z = np.dot(w,activation)+b
            zs.append(z)
            activation = sigmoid_vec(z)
            activations.append(activation)

        #backpropagation

        delta = self.cost_derivative(activations[-1],y)*sigmoid_prime_vec(zs[-1])
        self.reduceTraining(dataset,activation,delta)
        error_b[-1] = delta
        error_w[-1] = np.dot(delta,activations[-2].transpose())

        for l in range(2,self.numLayers):
            delta = np.dot(self.weights[-l+1].transpose(),delta)*sigmoid_prime_vec(zs[-l])
            error_b[-l] = delta
            error_w[-l] = np.dot(delta,activations[-l-1].transpose())

        return (error_w,error_b)

    def cost_derivative(self,activation,y):
        return 2*(activation-y)/(y*y)
        #return 2*(activation-y)

    def cost(self,activation,y):
        rd = 0
        for a,y in zip(activation,y):
            rd += np.power((y-a)/y,2)
            #rd += np.power(y-a,2)
        return rd/len(activation)

    def evaluate(self,dataset):
        rt = 0
        for x,y in dataset:
            rt += self.cost(self.feedforward(x),y)
        return np.sqrt(rt/len(dataset))

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid)

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

sigmoid_prime_vec = np.vectorize(sigmoid_prime)