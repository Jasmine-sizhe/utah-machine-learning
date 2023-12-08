from NN import *
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
    widths = [5, 10, 25, 50, 100]

    for i in range(2):
        if i==0:
            print("Train 50 epochs for random initialization")
        else:
            print("Train 50 epochs for zero initialization")
        for w in widths:
            model = NN(np.array(TrainAttributeData),np.array(TestAttributeData),TrainNumAttribute)
            model.Run(w,i)

    print("Pytorch Implementation")
    criterion = torch.nn.CrossEntropyLoss()
    for w in widths:
        for depths in [3,5,9]:
            for activ in ["RELU","TANH"]:
                print("widths: ",w," depths: ",depths," activation: ",activ)
                model = NN(np.array(TrainAttributeData), np.array(TestAttributeData), TrainNumAttribute)
                model.PytorchImplement(w, criterion, activeF=activ, numbersOfLayers=depths)