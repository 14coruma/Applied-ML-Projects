# This file includes code that I created in my first attempt at a NN
# for this XOR-type problem.
# The model seems to always converge to a model that predicts 0.5 for every
# input.
# I was unable to debug the program in a reasonable amount of time, so 
# I found/followed a stackoverflow discussion (https://stackoverflow.com/questions/36369335/xor-neural-network-converges-to-0-5)
# where someone was running into a similar issue.

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
from numba import njit
from copy import deepcopy

# A Perceptron Layer
class PLayer:
    weights = None # shape=(n_inputs+1 m_units)
    
    # Sigmoid
    phi = lambda s,V: np.array([1/(1+math.exp(-v)) for v in V])
    # Derivative of sigmoid
    dphi = lambda s,V: s.phi(V) * (1-s.phi(V))

    
    # Input: shape(n_inputs,1). Output: shape(m_weights,1), V
    def y(self, x):
        V = np.insert(x,0,1).dot(self.weights)
        return self.phi(V), V, np.insert(x,0,1)
    
    # Random weights and bias, initialized to be between -1 and 1
    def __init__(self, n_inputs, m_units):
        self.weights = np.random.rand(n_inputs+1, m_units)*2-1
        
    def __str__(self):
        return "Weights:\n{}\n".format(self.weights)


# A Multi-Layer Perceptron
class MLPerceptron:
    layers = [] # List of PLayers. layers[0] is input layer.

    # Input: shape(n_inputs, 1). Output: {0,1}, Vs
    def y(self, x, curr_layer = -1):
        if curr_layer == -len(self.layers):
            res = self.layers[0].y(x)
            return res[0], [res[1]], [res[2]]
        else:
            y_prev, V_prev, X_prev = self.y(x, curr_layer-1)
            res = self.layers[curr_layer].y(y_prev)
            V_prev.append(res[1])
            X_prev.append(res[2])
            return res[0], V_prev, X_prev
        
    def backprop(self, eta, err, Vs, Xs, curr_layer = 0):
        if curr_layer == len(self.layers)-1:
            # Delta rule for last layer
            delta = err * self.layers[-1].dphi(Vs[-1])
            shape = self.layers[-1].weights.shape
            dW = [(eta * delta * Xs[-1]).reshape(shape)]
            return delta, dW
        else:
            # Generalized delta rule
            delta_prev, dW = self.backprop(eta, err, Vs, Xs, curr_layer+1)
            delta = self.layers[curr_layer].dphi(Vs[curr_layer])
            delta = delta.reshape((1,self.layers[curr_layer].weights.shape[1]))
            val = np.sum([dk*wk for dk,wk in zip(delta_prev.T, self.layers[curr_layer+1].weights.T)], axis=0)
            val = val.reshape((self.layers[curr_layer+1].weights.shape[0],1))
            delta = val.dot(delta)
            delta_times_X = np.array([delta[i] * Xs[curr_layer][i] for i in range(len(Xs[curr_layer]))])
            dW.insert(0, eta * delta_times_X)
            return delta, dW
    
    def update_weights(self, dW):
        for i in range(len(self.layers)):
            self.layers[i].weights = self.layers[i].weights + dW[i]

        
    # shapes = [(n0,n1), (n1,n2), (n2,n3), ...]
    def __init__(self, shapes):
        self.layers = []
        for shape in shapes:
            self.layers.append(PLayer(shape[0], shape[1]))
            
    def __str__(self):
        str = ""
        for i in range(len(self.layers)):
            str += "Layer {}:\n{}\n".format(i, self.layers[i])
        return str

# Create a Multi-Layer Perceptron (with random weights and bias)
# Two layers. First has 4 inputs, 4 units. Second has 4 inputs, 1 unit.
p = MLPerceptron([(4,4), (4,1)])


# All 16 possible data points
X = [np.array(list(format(n, '#06b')[2:]), dtype=int) for n in range(16)]

# Train perceptron
epochs = 10000
eta = .75
all_mse = []
for epoch in range(epochs):
    # Randomize order of data
    random.shuffle(X)
    y = [sum(x)%2 for x in X]
    max_err, mse = 0, 0
    #dW_avg = None
    for x,y_true in zip(X,y):
        y_pred, Vs, Xs = p.y(x)
        err = y_true - y_pred
    
        _, dW = p.backprop(eta, err, Vs, Xs)
        p.update_weights(dW)
        #dW_avg = [x+np.array(y) for x,y in zip(dW_avg, dW)] if dW_avg is not None else dW
        
        if (abs(err) > max_err): max_err = abs(err)
        mse += err**2
    #p.update_weights([x/16 for x in dW_avg])
    
    all_mse.append(mse/16)
    if epoch%1000 == 0: print("epoch {}, mse: {}, max_abs_err: {}".format(epoch, all_mse[-1], max_err))
        
# Plot MSE
x = np.arange(0, epochs, 1)
y = np.array(all_mse)
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set(xlabel='epoch', ylabel='MSE',
       title='MSE at each epoch')
ax.grid()

fig.savefig("test.png")
plt.show()

# Randomize order of data
random.shuffle(X)
y = [sum(x)%2 for x in X]
max_err, mse = 0, 0
dW_avg, dB_avg = None, None
for x,y_true in zip(X,y):
    y_pred, Vs, Xs = p.y(x)
    print(x, y_pred)
