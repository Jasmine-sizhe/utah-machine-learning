import matplotlib.pyplot as plt
from BaggedTree import BaggedTrees, calculate_error_rate
import pandas as pd
from RandomForest import RandomForest

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

# Model training 
def train_Adaboost():
    train_data, test_data, attributes = load_bank_data()
    training_errors = []
    testing_errors = []

    return "A"


def train_Bagging():
    train_data, test_data, attributes = load_bank_data()
    training_errors = []
    testing_errors = []

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

import matplotlib.pyplot as plt

def train_RandomForest():
    train_data, test_data, attributes = load_bank_data()
    # Lists to store results
    trees_range = list(range(1, 11))
    feature_subsets = [2, 4, 6]
    results_train = {}
    results_test = {}

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
        ensemble_method = input('Ensemble method? a for Adabooost, b for Bagging, r for Random Forest\n')
        if ensemble_method=='a':
            train_Adaboost()
        if ensemble_method=='b':
            train_Bagging()
        else:
            train_RandomForest()
        print('\n')