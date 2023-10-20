import numpy as np
import pandas as pd
from numpy import linalg
import matplotlib.pyplot as plt
import random
from numpy.linalg import inv

random.seed(1)

'''
def gradient_descent(
    gradient, start, learn_rate, max_iter=50, tolerance=1e-06
):
    vector = start
    for _ in range(n_iter):
        diff = -learn_rate * gradient(vector)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
    return vector
'''

def BatchGradientDescent(x, y, r, tolerance=1e-5, max_iter=10000):
    #  initialize weight vector to be 0
    w = np.zeros(x.shape[1])
    b = 0
    iter = 0
    Cost = []
    for each_iter in range(max_iter):
        resd = y - (b + np.dot(x, w))
        J = 0.5 * np.sum(np.square(resd))
        new_b = b + r*(np.sum(resd))
        new_w = w + r*(np.dot((resd),x))
        b = new_b
        w = new_w
        diff = linalg.norm(w)
        Cost.append(J)
        iter += 1
        if (diff < tolerance) or (iter > max_iter):
            break
    return b, w, Cost

def StochasticGradientDescent(x, y, r, tolerance=1e-5, max_iter=10000):
    #  initialize weight vector to be 0
    w = np.zeros(x.shape[1])
    b = 0
    iter = 0
    Cost = []
    for t in range(max_iter):
        i = random.randint(0, len(train_x) - 1)
        resd1 = y - (b + np.dot(x, w))
        resd = y.loc[i] - (b + np.dot(x.loc[i], w))
        J = 0.5 * np.sum(np.square(resd1))
        new_b = b + r*(np.sum(resd))
        new_w = w + r*(np.dot((resd),x.loc[i]))
        b = new_b
        w = new_w
        diff = linalg.norm(w)
        Cost.append(J)
        iter += 1
        if (diff < tolerance) or (iter > max_iter):
            break
    return b, w, Cost