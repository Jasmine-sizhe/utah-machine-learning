import pandas as pd




if __name__ == "__main__":
    while True:
        dataset = input('Dataset? b for Bank, c for Car, e for exit\n')
        while dataset != 'b' and dataset != 'c' and dataset!='e':
            print("Sorry, unrecognized dataset")
            dataset = input('Dataset? b for Bank, c for Car\n')
        if dataset =='e':
            exit(0)
        maxDepth = int(input('Max depth of tree, input a number\n'))
        while maxDepth < 1:
            print("Sorry, max depth should greater than zero\n")
            maxDepth = int(input('Max depth of tree, input a number\n'))
        split = input('Split Algorithm? 0 for entropy, 1 for Majority error, 2 for gini index\n')
        while split != '0' and split != '1' and split != '2':
            print("Sorry, unrecognized split\n")
            split = input('Split Algorithm? 0 for entropy, 1 for Majority error, 2 for gini index\n')
        if dataset=='b':
            test_bank_dataset(maxDepth,int(split))
        else:
            Ttest_car_dataset(maxDepth,int(split))
        print('\n')