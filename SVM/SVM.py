import numpy as np
from scipy.optimize import minimize

class svm:
    TrainData = np.empty(1)
    TestData = np.empty(1)
    numAttribute = -1
    w = np.zeros(1)
    r = 0.01
    b = 1
    maxEpoch = 100

    def __init__(self, trainData, testData, numAttr):


        self.TrainData = np.array(trainData, dtype='float64')
        num_col = self.TrainData.shape[1]
        label = np.expand_dims(np.where(self.TrainData[:, -1] == 1, 1, -1),1)  # convert to 1/-1 for convient
        self.TrainData[:, num_col - 1] = 1
        self.TrainData = np.append(self.TrainData,label,1)

        self.TestData = np.array(testData, dtype='float64')
        label = np.expand_dims(np.where(self.TestData[:, -1] == 1, 1, -1), 1)  # convert to 1/-1 for convient
        self.TestData[:, num_col - 1] = 1
        self.TestData = np.append(self.TestData, label, 1)

        self.numAttribute = numAttr+1 #(add one for b)
        self.w = np.zeros(self.numAttribute)

    def TestError(self, data):
        x = np.array(data[:, 0:-1])
        y = np.matmul(x, self.w)
        y = np.where(y > 0, 1, -1)
        acc = np.count_nonzero(y == data[:, -1]) / len(data)
        return acc

    def runPrimal(self):
        C = [100/873,500/873,700/873]
        N = 873
        for c in C:
            w0 = np.zeros(self.numAttribute)
            self.w = np.zeros(self.numAttribute)
            for i in range(self.maxEpoch):
                # shuffule data
                np.random.shuffle(self.TrainData)
                for index in range(len(self.TrainData)):
                    error = (np.dot(self.w, self.TrainData[index][:-1]) + self.b) * self.TrainData[index][-1]
                    if error <= 1:
                        # update weights
                        self.w = self.w - self.r * w0 + self.r*c*N*self.TrainData[index][:-1] * self.TrainData[index][-1]
                    else:
                        w0 = (1-self.r)*w0
                self.updateLearningRate(i)
                trainError = self.TestError(self.TrainData)
                testError = self.TestError(self.TestData)
                print("C: ",c," Epoch ", i, " w: ", np.round(self.w,3), " Train Error: ", np.round(1-trainError,3), " Test Error: ", np.round(1-testError,3))
    def updateLearningRate(self,t):
        #for 2.a
        alpha = 0.02
        #self.r = 0.01/(1+0.01/alpha*t) # assume alpha = gamma = 0.01

        # for 2.b
        self.r = 0.01/(1+t)

    def runDual(self):
        C = [100/873,500/873,700/873]
        N = 873
        for c in C:
            num_data = train_x.shape[0]

            # set parameters
            x0 = np.random.rand(num_data)
            bnds = [(0, C)] * num_data
            Mat = np.zeros((num_data, num_data))

            def Mat_func(train_x, train_y):
                for i in range(num_data):
                    for j in range(num_data):
                        Mat[i, j] = np.dot(train_x[i], train_x[j]) * train_y[i] * train_y[j]
                return Mat

            Mat = Mat_func(train_x, train_y)
            # optimize
            res = minimize(main_function, x0, args=(Mat,), method='L-BFGS-B', jac=jac, bounds=bnds)

            # recover w, b
            w = np.sum([res.x[i] * train_y[i] * train_x[i, :] for i in range(train_x.shape[0])], axis=0)
            b = np.mean(train_y - np.dot(train_x, w))

            return w, b
            trainError = self.TestError(self.TrainData)
            testError = self.TestError(self.TestData)
            print("C: ",c," Epoch ", i, " w: ", np.round(self.w,3), " Train Error: ", np.round(1-trainError,3), " Test Error: ", np.round(1-testError,3))