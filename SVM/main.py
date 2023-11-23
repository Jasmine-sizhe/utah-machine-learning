from SVM import *
from Dual import *
# Take in file name return a dictionary
def readFiles(CSVfile):
    attributeData = []
    attributeLength = 0
    with open(CSVfile, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            attributeData.append(terms)
            attributeLength = len(terms) - 1

    return attributeData, attributeLength


if __name__ == "__main__":
    TrainAttributeData, TrainNumAttribute = readFiles("Data/bank-note/train.csv")
    TestAttributeData, TestNumAttribute = readFiles("Data/bank-note/test.csv")
    while 1:
        run = input(
            'Algorithm? 0 for Primal SVM, 1 for Dual SVM, 2 for Gaussian Kernel\n')
        while run != '0' and run != '1' and run!='2':
            print("Sorry, unrecognized Algorithm\n")
            run = input('Algorithm? 0 for Primal SVM, 1 for Dual SVM, 2 for Gaussian Kernel\n')
        if run == '0':
            primalSvm = svm(TrainAttributeData, TestAttributeData, TrainNumAttribute)
            primalSvm.runPrimal()
        elif run == '1':
            dualSvm = dualSVM(TrainAttributeData, TestAttributeData, TrainNumAttribute)
            dualSvm.runDual()
        else:
            dualSvm = dualSVM(TrainAttributeData, TestAttributeData, TrainNumAttribute)
            dualSvm.runDualGaussianKernel()
