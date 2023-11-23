import numpy as np
from scipy.optimize import minimize


class dualSVM:
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
        label = np.where(self.TrainData[:, -1] == 1, 1, -1)  # convert to 1/-1 for convient
        self.TrainData[:, num_col - 1] = label

        self.TestData = np.array(testData, dtype='float64')
        label = np.where(self.TestData[:, -1] == 1, 1, -1)  # convert to 1/-1 for convient
        self.TestData[:, num_col - 1] = label

        self.numAttribute = numAttr + 1  # (add one for b)
        self.w = np.zeros(self.numAttribute)

    def TestError(self, data):
        x = np.array(data[:, 0:-1])
        y = np.matmul(x, self.w) + self.b
        y = np.where(y > 0, 1, -1)
        acc = np.count_nonzero(y == data[:, -1]) / len(data)
        return acc

    def loss_function(self, alpha, Mat):
        return 0.5 * np.dot(alpha, np.dot(Mat, alpha)) - np.sum(alpha)
    def jac(self,alpha, Matrix):
        return np.dot(alpha.T, Matrix) - np.ones(alpha.shape[0])
    def runDual(self):
        C = [100 / 873, 500 / 873, 700 / 873]

        for c in C:
            # Initialize all the function parameters
            rows = self.TrainData.shape[0]
            bounds = [(0, c)] * rows
            Matrix = np.zeros((rows, rows))
            initialX = np.random.rand(rows)
            for i in range(rows):
                for j in range(rows):
                    Matrix[i, j] = np.dot(self.TrainData[:, 0:-1][i], self.TrainData[:, 0:-1][j]) * \
                                   self.TrainData[:, -1][
                                       i] * self.TrainData[:, -1][j]

            # call the minimize function
            result = minimize(self.loss_function, initialX, args=(Matrix,), method='SLSQP', jac=self.jac, bounds=bounds)

            # get w and b
            self.w = np.sum([result.x[i] * self.TrainData[:, -1][i] * self.TrainData[:, 0:-1][i, :] for i in
                             range(self.TrainData[:, 0:-1].shape[0])], axis=0)
            self.b = np.mean(self.TrainData[:, -1] - np.dot(self.TrainData[:, 0:-1], self.w))

            # Error checking
            trainError = self.TestError(self.TrainData)
            testError = self.TestError(self.TestData)
            print("C: ", c, " w: ", np.round(self.w, 3), " Train Error: ", np.round(1 - trainError, 3), " Test Error: ",
                  np.round(1 - testError, 3))

    def kernelTestError(self, Data, rows, alpha, b, g):

        Kernel = np.zeros((self.TrainData.shape[0], rows))
        for i in range(self.TrainData.shape[0]):
            for j in range(rows):
                Kernel[i, j] = np.exp(
                    (-1) * np.linalg.norm(self.TrainData[:, 0:-1][i] - Data[:, 0:-1][j], 2) / g)
        Kernel = Kernel * np.reshape(alpha, (alpha.shape[0], 1)) * np.reshape(self.TrainData[:, -1],
                                                                              (self.TrainData[:, -1].shape[0], 1))

        Acc = 0
        for i in range(rows):
            predict = np.sign((np.sum(Kernel, axis=0) + b)[i])
            if predict == Data[:, -1][i]:
                Acc += 1
        return Acc / rows

    def runDualGaussianKernel(self):
        C = [500 / 873]
        Gamma = [0.1, 0.5, 1, 5, 100]
        rows = self.TrainData.shape[0]

        for c in C:
            sup_vec = np.zeros(rows)
            for g in Gamma:
                # Initialize all the function parameters
                bounds = [(0, c)] * rows
                Matrix = np.zeros((rows, rows))
                initialX = np.random.rand(rows)
                for i in range(rows):
                    for j in range(rows):
                        Matrix[i, j] = np.exp(
                            (-1) * np.linalg.norm(self.TrainData[:, 0:-1][i] - self.TrainData[:, 0:-1][j], 2) / g) * \
                                       self.TrainData[:, -1][
                                           i] * self.TrainData[:, -1][j]

                # call the minimize function
                Opresult = minimize(self.loss_function, initialX, args=(Matrix,), method='SLSQP', jac=self.jac,
                                    bounds=bounds)

                # Calculate the kernel
                Kernel = np.zeros((rows, rows))
                for i in range(rows):
                    for j in range(rows):
                        Kernel[i, j] = np.exp(
                            (-1) * np.linalg.norm(self.TrainData[:, -1][i] - self.TrainData[:, 0:-1][j], 2) / g)
                Kernel *= Opresult.x * self.TrainData[:, -1]

                self.w = np.sum(
                    [Opresult.x[i] * self.TrainData[:, -1][i] * self.TrainData[:, 0: -1][i, :] for i in range(rows)],
                    axis=0)
                self.b = np.mean(self.TrainData[:, -1] - np.sum(Kernel, axis=0))

                # Error checking
                trainError = self.kernelTestError(self.TrainData, rows, Opresult.x, self.b, g)
                testError = self.kernelTestError(self.TestData, self.TestData.shape[0], Opresult.x, self.b, g)
                supportVector = np.sum(Opresult.x != 0.0)
                print("Gamma: ", g, " C: ", c, " w: ", np.round(self.w, 3), " Train Error: ",
                      np.round(1 - trainError, 5), " Test Error: ",
                      np.round(1 - testError, 5), " support Vector: ", supportVector)
                if c == 500 / 873:
                    intersect = len(np.intersect1d(sup_vec, np.argwhere(Opresult.x > 0)))
                    print("g:", g, " intersect: ", intersect)
                    sup_vec = np.argwhere(Opresult.x > 0)


'''
Part (a)
C:  0.1145475372279496  w:  [-1.24  -0.673 -0.73  -0.287]  Train Error:  0.034  Test Error:  0.038
C:  0.572737686139748  w:  [-1.706 -0.853 -0.994 -0.307]  Train Error:  0.036  Test Error:  0.036
C:  0.8018327605956472  w:  [-1.708 -0.854 -0.994 -0.308]  Train Error:  0.036  Test Error:  0.036

Part (b and c)
Gamma:  0.1  C:  0.1145475372279496  w:  [-204.633 -280.244   45.186   11.615]  Train Error:  0.0  Test Error:  0.442  support Vector:  872
Gamma:  0.5  C:  0.1145475372279496  w:  [-204.59  -280.135   45.144   11.569]  Train Error:  0.029  Test Error:  0.162  support Vector:  872
Gamma:  1  C:  0.1145475372279496  w:  [-155.651 -222.208   41.412    8.944]  Train Error:  0.033  Test Error:  0.084  support Vector:  866
Gamma:  5  C:  0.1145475372279496  w:  [-47.813 -53.933   7.257  -1.428]  Train Error:  0.029  Test Error:  0.042  support Vector:  489
Gamma:  100  C:  0.1145475372279496  w:  [-145.18  -137.216  -17.618  -11.779]  Train Error:  0.157  Test Error:  0.168  support Vector:  838
Gamma:  0.1  C:  0.572737686139748  w:  [-1023.776 -1394.84    231.227    49.432]  Train Error:  0.0  Test Error:  0.398  support Vector:  872
gamma: 0.1  #intersect:  1
Gamma:  0.5  C:  0.572737686139748  w:  [-567.162 -887.632  177.876   49.509]  Train Error:  0.0  Test Error:  0.016  support Vector:  860
gamma: 0.5  #intersect:  860
Gamma:  1  C:  0.572737686139748  w:  [-216.693 -337.422   83.279   15.064]  Train Error:  0.0  Test Error:  0.0  support Vector:  868
gamma: 1  #intersect:  856
Gamma:  5  C:  0.572737686139748  w:  [-51.846 -57.438   1.623   3.251]  Train Error:  0.0  Test Error:  0.002  support Vector:  465
gamma: 5  #intersect:  463
Gamma:  100  C:  0.572737686139748  w:  [-245.439 -175.951  -95.019   -4.919]  Train Error:  0.083  Test Error:  0.104  support Vector:  389
gamma: 100  #intersect:  236
Gamma:  0.1  C:  0.8018327605956472  w:  [-1429.191 -1938.017   326.953    61.063]  Train Error:  0.0  Test Error:  0.38  support Vector:  872
Gamma:  0.5  C:  0.8018327605956472  w:  [-582.309 -922.209  183.012   56.086]  Train Error:  0.0  Test Error:  0.014  support Vector:  861
Gamma:  1  C:  0.8018327605956472  w:  [-216.727 -336.881   80.375   15.866]  Train Error:  0.0  Test Error:  0.0  support Vector:  849
Gamma:  5  C:  0.8018327605956472  w:  [-52.034 -57.944   0.981   4.487]  Train Error:  0.0  Test Error:  0.0  support Vector:  584
Gamma:  100  C:  0.8018327605956472  w:  [-259.747 -185.833 -110.371   -0.728]  Train Error:  0.081  Test Error:  0.094  support Vector:  801



'''