import pandas as pd
import sys
sys.path.append("../Decision Tree")
from TreeNode import TreeNode
from DecisionTree import ID3
import math
import matplotlib.pyplot as plt

from utils import predict, calculate_error_rate, preprocess_numerical_columns

# Get Decision Stump
def weighted_entropy(data, weights, class_list):
    total = sum(weights)
    entropy = 0
    for label in class_list:
        weight_for_label = sum(w for w, d in zip(weights, data.iloc[:, -1]) if d == label)
        p = weight_for_label / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def calculate_info_gain(feature_name, data, weights, class_list):
    total_row = data.shape[0]
    feature_info = 0.0
    total_entropy = weighted_entropy(data, weights, class_list)

    for feature_value in data[feature_name].unique():
        feature_value_data = data[data[feature_name] == feature_value]  # filtering rows with that feature_value
        feature_value_weights = [w for w, d in zip(weights, data[feature_name]) if d == feature_value]
        feature_value_probability = sum(feature_value_weights) / sum(weights)
        
        feature_value_entropy = weighted_entropy(feature_value_data, feature_value_weights, class_list)
        feature_info += feature_value_probability * feature_value_entropy

    return total_entropy - feature_info


def find_best_attribute(data, attributes, weights, class_list):
    max_info_gain = -1
    max_info_feature = None

    for attribute in attributes:  #for each feature in the dataset
        feature_info_gain = calculate_info_gain(attribute, data, weights, class_list)
        if max_info_gain < feature_info_gain: #selecting feature name with highest information gain
            max_info_gain = feature_info_gain
            max_info_feature = attribute
        # print(f"Attribute: {attribute}, Info Gain: {feature_info_gain}, Weights: {weights}")
            
    return max_info_feature

def weighted_decision_stump(data, attributes, weights):
    class_list = data.iloc[:, -1].unique().tolist()

    best_attribute = find_best_attribute(data, attributes, weights, class_list)
    root = TreeNode(attributes=best_attribute)
    # print(f"Selected Best Attribute: {best_attribute}")
    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value]
        most_common_label = subset.iloc[:, -1].value_counts().idxmax()
        # print(f"Value: {value}, Most Common Label: {most_common_label}")
        
        child_node = TreeNode(label=most_common_label, attributes=value) 
        root.add_child(child_node)

    return root


import numpy as np

def adaboost(data, attributes, T):
    """
    data: DataFrame containing the training data
    attributes: List of column names of the features
    T: Number of iterations or weak classifiers to train
    """
    
    N = len(data)  # Number of data points
    weights = [1/N] * N  # Initialize weights
    weak_classifiers = []  # List to store the trained weak classifiers (decision stumps)
    alphas = []  # List to store the alpha values for each classifier
    
    for t in range(T):
        # Train a decision stump using the weighted data
        # print("weights before tree grow: ", weights)
        stump = weighted_decision_stump(data, attributes, weights)
        weak_classifiers.append(stump)
        
        # Make predictions using the stump on all data points
        predictions = predict(stump, data)  # assuming you have a 'predict' function
        actual = data.iloc[:, -1].tolist()
        # print("predictions: ", predictions)
        # print("actual: ", actual)
        
        # Calculate the weighted error
        weighted_error = sum([weights[i] for i in range(N) if predictions[i] != actual[i]])
        print(f"Iteration {t + 1} weighted_error: {weighted_error}")
        # Calculate the alpha value for this classifier
        alpha = 0.5 * np.log((1 - weighted_error) / max(weighted_error, 1e-10))
        alphas.append(alpha)
        print(f"Iteration {t + 1} alphas: {alphas}")
        # Update the weights for the next iteration
        for i in range(N):
            if predictions[i] == actual[i]:
                weights[i] = weights[i] * np.exp(-alpha)
                # print("true label for {} and weights {}".format(i, weights[i]))
            else:
                weights[i] = weights[i] * np.exp(alpha)
                # print("false label for {} and weights {}".format(i, weights[i]))

        # print("weights before normalization: ", weights)
        
        # Normalize the weights to sum up to 1
        total_weight = sum(weights)
        # print("total_weight",total_weight)
        weights = [w/total_weight for w in weights]

        # print("sum weights",sum(weights))
        # print(f"Iteration {t + 1} weights: {weights}")

    
    return weak_classifiers, alphas

def adaboost_predict(data_point, weak_classifiers, alphas):
    # predictions = [predict(classifier, data_point) for classifier in weak_classifiers]
    predictions = [predict(classifier, pd.DataFrame([data_point])) for classifier in weak_classifiers]
    weighted_predictions = [alphas[i] * (1 if predictions[i] == 'yes' else -1) for i in range(len(alphas))]
    return 'yes' if sum(weighted_predictions) > 0 else 'no'


