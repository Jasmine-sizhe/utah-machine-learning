import pandas as pd
import numpy as np
def predict(tree, df_test):
    """
    tree: The tree that has been build from ID3
    df_test: test dataset, dataframe
    return: a list of all prediction labels
    """
    predictions = []
    for index, row in df_test.iterrows():
        node = tree
        while node.children: 
            attribute_name = node.attributes 
            attribute_value = row[attribute_name] 
            matched_child = None
            for child in node.children:
                if child.attributes == attribute_value:  
                    matched_child = child  
                    break
            if matched_child:
                node = matched_child
                for subnode in node.children:
                    node = subnode
            else:
                break
        predictions.append(node.label)  

    return predictions


def calculate_error_rate(predictions, true_labels):
    """
    predictions: list of prediction labels using ID3
    true_labels: real labels
    return: error_rate
    """
    if len(predictions) != len(true_labels):
        raise ValueError("Number of predictions and true label do not match")

    incorrect_predictions = 0
    total_samples = len(predictions)

    for i in range(total_samples):
        if predictions[i] != true_labels[i]:
            incorrect_predictions += 1

    error_rate = incorrect_predictions / total_samples
    return error_rate

# Preprocessing needed for bank dataset
def preprocess_numerical_columns(df, numerical_columns):
    # Covert numerical value to categorical value
    df_processed = df.copy()

    # Deal with unknown in numerical columns
    for column in numerical_columns:
        df_processed[column] = df_processed[column].replace('unknown', np.nan)

    # Calculate medians with exsiting numerical value in numerical columns
    medians = df_processed[numerical_columns].median(skipna=False)

    # Assign Low and High to replace numerical values
    for column in numerical_columns:
        df_processed[column] = pd.cut(df_processed[column], bins=[float("-inf"), medians[column], float("inf")], labels=["Low", "High"])
    for column in numerical_columns:
        df_processed[column] = df_processed[column].cat.add_categories("unknown").fillna("unknown")

    return df_processed

# Data processing with unknown value in the datasets (Bank dataset)
# If imputation needed, need to call this function before feed into the ID3 algorithm
def fill_unknown(df):
    """
    Input is dataframe, return is also a dataframe with filed unknown dataset
    example: df_filled_train = fill_unknown(df_all_categorical_bank_train)
    Then use df_filled_train feed into ID3
    """
    df_res = df.copy()
    for col in df_res.columns:
        unique, pos = np.unique(df_res[col], return_inverse=True)  
        counts = np.bincount(pos) 
        maxpos = counts.argmax()  
        df_res[col] = np.where(df_res[col] == "unknown", unique[maxpos], df[col]) 
    return df_res