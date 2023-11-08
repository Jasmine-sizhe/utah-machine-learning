from perceptron import *
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
    TrainAttributeData, TrainNumAttribute = readFiles("Data/train.csv")
    TestAttributeData, TestNumAttribute = readFiles("Data/test.csv")

    while 1:
        run = input(
            'Algorithm? 0 for Standard, 1 for Voted, 2 for Average\n')
        while run != '0' and run != '1' and run != '2':
            print("Sorry, unrecognized Algorithm\n")
            run = input('Algorithm? 0 for Standard, 1 for Voted, 2 for Average\n')
        perceptronModel = Perceptron(TrainAttributeData, TestAttributeData, TrainNumAttribute)
        if run == '0':
            perceptronModel.runStandard()
        elif run == '1':
            perceptronModel.runVoted()
        else:
            perceptronModel.runAverage()
