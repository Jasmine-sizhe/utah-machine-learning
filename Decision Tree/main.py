import pandas as pd
from DecisionTree import ID3
from utils import predict, calculate_error_rate, preprocess_numerical_columns


def test_bank_dataset(max_depth,purity_method):
    # Loading data and read as dataframe
    test_file_path = "Data/bank-4/test.csv"
    train_file_path = "Data/bank-4/train.csv"
    column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
    df_bank_train = pd.read_csv(train_file_path, names=column_names)
    df_bank_test = pd.read_csv(test_file_path, names=column_names)
    bank_attributes = df_bank_train.columns.tolist()[:-1]
    
    # Preprocessing needed for bank dataset
    numerical_columns = ["default", "contact", "month", "duration", "campaign", "pdays"]
    df_all_categorical_bank_train = preprocess_numerical_columns(df_bank_train, numerical_columns)
    df_all_categorical_bank_test = preprocess_numerical_columns(df_bank_test, numerical_columns)

    # Run Tree
    root = ID3(df_all_categorical_bank_train, bank_attributes, max_depth=max_depth, purity_measurement=purity_method)
    predictions = predict(root, df_all_categorical_bank_train)
    true_labels = df_all_categorical_bank_train.iloc[:, -1].tolist()
    error_rate = calculate_error_rate(predictions, true_labels)
    print(f"Max Depth {max_depth}: Training Error Rate {error_rate:.3f}")

    root = ID3(df_all_categorical_bank_train, bank_attributes, max_depth=max_depth, purity_measurement=purity_method)
    predictions = predict(root, df_all_categorical_bank_test)
    true_labels = df_all_categorical_bank_test.iloc[:, -1].tolist()
    error_rate = calculate_error_rate(predictions, true_labels)
    print(f"Max Depth {max_depth}: Testing Error Rate {error_rate:.3f}")


def test_car_dataset(max_depth, purity_method):
    # Read data
    test_file_path = "Data/car-4/test.csv"
    train_file_path = "Data/car-4/train.csv"
    column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
    df_car_train = pd.read_csv(train_file_path, names=column_names)
    df_car_test = pd.read_csv(test_file_path, names=column_names)
    car_attributes = df_car_train.columns.tolist()[:-1]

    # Run Tree
    root = ID3(df_car_train, car_attributes, max_depth=max_depth, purity_measurement=purity_method)
    predictions = predict(root, df_car_train)
    true_labels = df_car_train.iloc[:, -1].tolist()
    error_rate = calculate_error_rate(predictions, true_labels)
    print(f"Max Depth {max_depth}: Training Error Rate {error_rate:.3f}")

    root = ID3(df_car_train, car_attributes, max_depth=max_depth, purity_measurement=purity_method)
    predictions = predict(root, df_car_test)
    true_labels = df_car_test.iloc[:, -1].tolist()
    error_rate = calculate_error_rate(predictions, true_labels)
    print(f"Max Depth {max_depth}: Testing Error Rate {error_rate:.3f}")

    

if __name__ == "__main__":
    while True:
        dataset = input('Dataset? b for Bank, c for Car, e for exit\n')
        while dataset != 'b' and dataset != 'c' and dataset!='e':
            print("Sorry, unrecognized dataset")
            dataset = input('Dataset? b for Bank, c for Car\n')
        if dataset =='e':
            exit(0)
        max_depth = int(input('Max depth of tree, input a number\n'))
        while max_depth < 1:
            print("Sorry, max depth should greater than zero\n")
            max_depth = int(input('Max depth of tree, input a number\n'))
        purity_method = input('purity method? entropy, majority_error, gini?')
        while purity_method != 'entropy' and purity_method != 'majority_error' and purity_method != 'gini':
            print("Sorry, unrecognized split\n")
            purity_method = input('purity method? entropy, majority_error, gini?\n')
        if dataset=='b':
            test_bank_dataset(max_depth,int(purity_method))
        else:
            test_car_dataset(max_depth,int(purity_method))
        print('\n')