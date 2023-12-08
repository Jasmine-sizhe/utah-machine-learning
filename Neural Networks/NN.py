import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable


def MoveTowardGradient(w, gradients, lr):
    for i in range(1, len(w) // 2 + 1):
        w['W' + str(i)] -= lr * gradients['dW' + str(i)]
        w['b' + str(i)] = w['b' + str(i)] - lr * gradients['db' + str(i)]
    return w


def lr_schedule(r0, t):
    return r0 / (1 + 1 * r0 * t)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NN:
    TrainData = np.empty(1)
    TestData = np.empty(1)
    TrainLabel = np.empty(1)
    TestLabel = np.empty(1)
    numAttribute = -1
    lr = 0.01
    maxEpoch = 50

    def __init__(self, trainData, testData, numAttr):
        self.TrainData = np.array(trainData[:, :-1], dtype='float64').T
        self.TrainLabel = np.expand_dims(trainData[:, -1], 1).T  # convert to 1/-1 for convient
        self.TrainLabel = self.TrainLabel.astype(int)
        self.TestData = np.array(testData[:, :-1], dtype='float64').T
        self.TestLabel = np.expand_dims(testData[:, -1], 1).T  # convert to 1/-1 for convient
        self.TestLabel = self.TestLabel.astype(int)
        self.numAttribute = numAttr + 1  # (add one for b)

    def Train(self, widths=5, initialization=0):
        # default append two hidden layers to the weights
        Layers = []
        for i in range(2):
            Layers.append(widths)

        x = self.TrainData
        y = self.TrainLabel
        # if initialization is true then initialize all zeros
        # else initialize randomly
        if initialization:
            weights = dict()
            for i in range(len(Layers)):
                weights['W' + str(i + 1)] = np.zeros((Layers[i], x.shape[0] if i==0 else Layers[i - 1]))
                weights['b' + str(i + 1)] = np.zeros((Layers[i], 1))
            #  for the last layer
            weights['W3'] = np.zeros((y.shape[0], Layers[-1]))
            weights['b3'] = np.zeros((y.shape[0], 1))

        else:
            weights = dict()
            for i in range(len(Layers)):
                num_hidden = Layers[i]
                weights['W' + str(i + 1)] = np.random.normal(0, 1, (num_hidden, x.shape[0] if i == 0 else Layers[i - 1]))
                weights['b' + str(i + 1)] = np.zeros((num_hidden, 1))
            #  for the last layer
            weights['W3'] = np.random.normal(0, 1, (y.shape[0], Layers[-1]))
            weights['b3'] = np.zeros((y.shape[0], 1))
        num = x.shape[1]
        idx = np.arange(num)
        for epoch in range(self.maxEpoch):
            # reshuffle the training samples
            np.random.shuffle(idx)
            x = x[:, idx]
            y = y[:, idx]

            # SGD
            for i in range(num):
                # forward pass
                predict, cache = self.Forward(x[:, i], weights)
                # backward pass
                grad_weights = self.Backward(y[:, i], predict, weights, cache)
                self.lr = lr_schedule(self.lr, epoch)
                # update the weights
                weights = MoveTowardGradient(weights, grad_weights, self.lr)
        return weights

    def Backward(self, y, predict, w, cache):
        gradients = {}
        deltaZ = predict - y
        for i in reversed(range(len(w) // 2)):
            gradients['dW' + str(i + 1)] = np.dot(deltaZ, cache['A' + str(i)].T)
            gradients['db' + str(i + 1)] = np.sum(deltaZ, axis=1)

            dA_prev = np.dot(w['W' + str(i + 1)].T, deltaZ)
            deltaZ = dA_prev * sigmoid(cache['Z' + str(i)]) * (1 - sigmoid(cache['Z' + str(i)]))

        return gradients

    def Forward(self, x, W):
        cache = dict()

        A = x[:, np.newaxis]
        cache['Z0'] = x[:, np.newaxis]
        cache["A0"] = x[:, np.newaxis]

        # number of layers in the neural network
        for i in range(len(W) // 2):
            w = W['W' + str(i + 1)]
            Z = np.dot(w, A)
            A = sigmoid(Z)

            cache['Z' + str(i + 1)] = Z
            cache['A' + str(i + 1)] = A
        return A, cache

    def PredictAndTest(self, x, w, y):
        A = x
        for i in range(len(w) // 2):
            W = w['W' + str(i + 1)]
            Z = np.dot(W, A)
            A = sigmoid(Z)

        return 1 - np.sum(y == np.where(A >= 0.5, 1, 0)) / y.shape[1]

    def Run(self, widths=5, initiliaze=0):
        weights = self.Train(widths, initiliaze)
        train_error = self.PredictAndTest(self.TrainData, weights, self.TrainLabel)
        test_error = self.PredictAndTest(self.TestData, weights, self.TestLabel)
        print("Widths: ", widths, 'Train Error', train_error, 'Test Error', test_error)

    def PytorchImplement(self, widths, criterion, activeF="RELU", numbersOfLayers=3):
        # Define the three-layer neural network

        layers = []
        for i in range(numbersOfLayers):
            inputs = widths
            outputs = widths
            if i == 0:
                inputs = self.TrainData.shape[0]
            if i == numbersOfLayers - 1:
                outputs = 2
            layers.append(nn.Linear(inputs, outputs))
            if activeF == "RELU":
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Tanh())
        model = nn.Sequential(*layers)
        optimizer = torch.optim.Adam(model.parameters())
        if activeF == "RELU":
            # Initialize the weights using Xavier initialization
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)
        else:
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_uniform(m.weight)

        # Finish initiliaze model parameter

        model.train()
        model.float()

        self.TrainData = torch.from_numpy(self.TrainData.T).float()
        self.TrainLabel = torch.squeeze(torch.from_numpy(self.TrainLabel.T)).float()
        self.TestData = torch.from_numpy(self.TestData.T).float()
        self.TestLabel = torch.squeeze(torch.from_numpy(self.TestLabel.T)).float()

        train_data = TensorDataset(self.TrainData, self.TrainLabel)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_data = TensorDataset(self.TestData, self.TestLabel)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
        # Iterate through the number of epochs
        for epoch in range(self.maxEpoch):
            # Iterate through the data
            correct = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = Variable(data), Variable(target)
                target = target.long()
                # Reset the gradients
                optimizer.zero_grad()

                # Forward pass
                output = model(data)

                # Compute the loss
                loss = criterion(output, target)

                # Backward pass
                loss.backward()

                # Update the weights
                optimizer.step()

        # Finish Training
        model.eval()
        total_loss = 0
        with torch.no_grad():
            correct = 0
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = Variable(data), Variable(target)
                target = target.long()
                output = model(data)
                output = torch.argmax(output, 1)
                correct += (output == target).float().sum()
            TestAccuracy = 100 * correct / len(self.TestData)

        with torch.no_grad():
            correct = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = Variable(data), Variable(target)
                target = target.long()
                output = model(data)
                output = torch.argmax(output, 1)
                correct += (output == target).float().sum()
            TrainAccuracy = 100 * correct / len(self.TrainData)
        print("Train Accuracy = {}".format(TrainAccuracy), "Test Accuracy = {}".format(TestAccuracy))