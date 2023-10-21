import matplotlib.pyplot as plt
from BaggedTree import BaggedTrees, calculate_error_rate, run_bagged_trees_experiment
import pandas as pd
from RandomForest import RandomForest
from sklearn.model_selection import train_test_split
from Adaboost import adaboost, adaboost_predict

# Loading and preprocessing data
def preprocess_data(df):
    # Convert continuous attributes to binary
    for column in ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']:
        median = df[column].median()
        df[column] = df[column].apply(lambda x: 1 if x > median else 0)
    
    # Note: For columns with "unknown", we'll leave them as is. Pandas will treat them as a separate category.
    
    return df

def load_bank_data():
    # Load the training and test data
    test_file_path = "Data/bank-4/test.csv"
    train_file_path = "Data/bank-4/train.csv"
    column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
    
    df_bank_train = pd.read_csv(train_file_path, names=column_names)
    df_bank_test = pd.read_csv(test_file_path, names=column_names)
    bank_attributes = df_bank_train.columns.tolist()[:-1]

    # Apply preprocessing to train and test datasets (assuming you have already defined preprocess_data)
    train_data = preprocess_data(df_bank_train)
    test_data = preprocess_data(df_bank_test)
    attributes = bank_attributes

    return train_data, test_data, attributes

def load_credit_card_data():
    file_path="Data/credit-card/credit_card.xls"
    df = pd.read_excel(file_path, header=1)

    # Deal with numeric value except target variable
    for column in df.columns[:-1]:
        if df[column].dtype == 'int64':
            median_value = df[column].median()
            df[column] = (df[column] >= median_value).astype(int)

    train_data, test_data = train_test_split(df, train_size=24000, test_size=6000, random_state=42)
    attributes = train_data.columns.tolist()[:-1]

    return train_data, test_data, attributes

# Model training 
def train_Adaboost(dataset_alias):
    print("-----starting loading data for adaboost-------")
    if dataset_alias == 'b':
        train_data, test_data, attributes = load_bank_data()
    elif dataset_alias == 'c':
        train_data, test_data, attributes = load_credit_card_data()
    else:
        print("Unrecognized dataset alias!")
        return None
    print("-----starting training for adaboost-------")
    train_errors = []
    test_errors = []
    T_values = list(range(1, 11))

    for T in T_values:
        weak_classifiers, alphas = adaboost(train_data, attributes, T)

        train_predictions = [adaboost_predict(row[1], weak_classifiers, alphas) for row in train_data.iterrows()]
        mismatches_train = sum([pred != actual for pred, actual in zip(train_predictions, train_data["y"].tolist())])
        train_error = mismatches_train / len(train_data)
        train_errors.append(train_error)

        test_predictions = [adaboost_predict(row[1], weak_classifiers, alphas) for row in test_data.iterrows()]
        mismatches_test = sum([pred != actual for pred, actual in zip(test_predictions, test_data["y"].tolist())])
        test_error = mismatches_test / len(test_data)
        test_errors.append(test_error)

    plt.figure(figsize=(10,6))
    plt.plot(T_values, train_errors, label="Train Error", marker='o')
    plt.plot(T_values, test_errors, label="Test Error", marker='o')
    plt.xlabel("Number of Iterations (T)")
    plt.ylabel("Error")
    plt.title("Training and Test Errors as T varies")
    plt.legend()
    plt.grid(True)
    plt.show()



def train_Bagging(dataset_alias):
    print("-----starting loading data for bagging-------")
    if dataset_alias == 'b':
        train_data, test_data, attributes = load_bank_data()
    elif dataset_alias == 'c':
        train_data, test_data, attributes = load_credit_card_data()
    else:
        print("Unrecognized dataset alias!")
        return None
    training_errors = []
    testing_errors = []

    print("-----starting training for bagging-------")
    for n in range(1, 11):  # Looping n from 1 to 10
        bagged_model = BaggedTrees(n_trees=n)
        bagged_model.fit(train_data, attributes)

        # Training error
        predictions = bagged_model.predict(train_data)
        true_labels_train = train_data.iloc[:, -1].tolist()
        error_rate_train = calculate_error_rate(predictions, true_labels_train)
        training_errors.append(error_rate_train)

        # Testing error
        predictions = bagged_model.predict(test_data)
        true_labels_test = test_data.iloc[:, -1].tolist()
        error_rate_test = calculate_error_rate(predictions, true_labels_test)
        testing_errors.append(error_rate_test)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), training_errors, label='Training Error', marker='o')
    plt.plot(range(1, 11), testing_errors, label='Testing Error', marker='x')
    plt.xlabel('Number of Trees')
    plt.ylabel('Error Rate')
    plt.title('Error Rates vs. Number of Trees')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, 11))
    plt.show()
    # Run bagged tree
    print("-------run bagged trees experiment---------")
    run_bagged_trees_experiment(train_data, test_data, attributes)

def train_RandomForest(dataset_alias):
    print("-----starting loading data for random forest-------")
    if dataset_alias == 'b':
        train_data, test_data, attributes = load_bank_data()
    elif dataset_alias == 'c':
        train_data, test_data, attributes = load_credit_card_data()
    else:
        print("Unrecognized dataset alias!")
        return None
    
    # Lists to store results
    trees_range = list(range(1, 11))
    feature_subsets = [2, 4, 6]
    results_train = {}
    results_test = {}

    print("-----starting training for random forest-------")
    for feature_subset in feature_subsets:
        error_rates_train = []
        error_rates_test = []
        for n_trees in trees_range:
            print(f"Training Random Forest with {n_trees} trees and feature subset size {feature_subset} ...")
            
            # Initialize and train RandomForest
            rf_model = RandomForest(n_trees, feature_subset)
            rf_model.fit(train_data, attributes)
            
            # Predict and calculate error rate on training data
            predictions_train = rf_model.predict(train_data)
            true_labels_train = train_data.iloc[:, -1].tolist()
            error_rate_train = calculate_error_rate(predictions_train, true_labels_train)
            error_rates_train.append(error_rate_train)
            
            # Predict and calculate error rate on test data
            predictions_test = rf_model.predict(test_data)
            true_labels_test = test_data.iloc[:, -1].tolist()
            error_rate_test = calculate_error_rate(predictions_test, true_labels_test)
            error_rates_test.append(error_rate_test)
            
            print(f"Training Error Rate: {error_rate_train} | Testing Error Rate: {error_rate_test}\n")
        
        results_train[feature_subset] = error_rates_train
        results_test[feature_subset] = error_rates_test

    # Plotting the results
    for feature_subset in feature_subsets:
        plt.plot(trees_range, results_train[feature_subset], '-o', label=f"Feature Subset: {feature_subset} (Train)")
        plt.plot(trees_range, results_test[feature_subset], '--x', label=f"Feature Subset: {feature_subset} (Test)")

    plt.xlabel("Number of Trees")
    plt.ylabel("Error Rate")
    plt.title("Random Forest Error Rate vs. Number of Trees")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    while True:
        dataset = input('Dataset? b for Bank Dataset, c for Credit-card dataset, e for exit\n')
        while dataset != 'b' and dataset != 'c' and dataset!='e':
            print("Sorry, unrecognized dataset")
            dataset = input('Dataset? b for Bank Dataset, c for Credit-card dataset\n')
        if dataset =='e':
            exit(0)
        if dataset=='b':
            train_Adaboost('b')
            train_Bagging('b')       
            train_RandomForest('b')
        else:
            train_Adaboost('c')
            train_Bagging('c')
            train_RandomForest('c')
        print('\n')