import pandas as pd
import numpy as np
from LinearRegression import BatchGradientDescent, StochasticGradientDescent
import matplotlib.pyplot as plt

def test_concrete_bgd():
    print("-----starting bgd -------")
    train = pd.read_csv('Data/train.csv', header=None)
    test = pd.read_csv('Data/test.csv', header=None)
    print("-----read files done -------")
    train_x = train.iloc[:, 0:(len(train.columns)-1)]
    train_y = train.iloc[:, -1]
    test_x = test.iloc[:, 0:(len(test.columns)-1)]
    test_y = test.iloc[:, -1]
    print("----- start bgd func -------")
    b1, w1, Cost1 = BatchGradientDescent(train_x, train_y, 0.01)
    resd = test_y - (b1 + np.dot(test_x, w1))
    J1 = 0.5 * np.sum(np.square(resd))
    print("b:", b1, "w:", w1, "cost: ", J1)

    plt.plot(Cost1)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()

def test_concrete_sgd():
    train = pd.read_csv('Data/train.csv', header=None)
    test = pd.read_csv('Data/test.csv', header=None)
    train_x = train.iloc[:, 0:(len(train.columns)-1)]
    train_y = train.iloc[:, -1]
    test_x = test.iloc[:, 0:(len(test.columns)-1)]
    test_y = test.iloc[:, -1]

    b2, w2, Cost2 = StochasticGradientDescent(train_x, train_y, 0.001)
    resd = test_y - (b2 + np.dot(test_x, w2))
    J2 = 0.5 * np.sum(np.square(resd))
    print("b:", b2, "w:", w2, "cost: ", J2)

    plt.plot(Cost2)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()

if __name__ == "__main__":
    while True:
        optimization_method = input('Optimization Method? bgd for Batch Gradient Descent, sgd for Stochastic Gradient Descent, e for exit\n')
        while optimization_method != 'bgd' and optimization_method != 'sgd':
            print("Sorry, unrecogonized optimization method\n")
            optimization_method = input('Optimization Method? bgd for Batch Gradient Descent, sgd for Stochastic Gradient Descent, e for exit\n')
        if optimization_method =='e':
            exit(0)
        if optimization_method=='bgd':
            test_concrete_bgd()
        else:
            test_concrete_sgd()
        print('\n')