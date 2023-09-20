import pandas as pd
import numpy as np
import math
import TreeNode

# Find most common label
def find_most_common_label(labels):
    label_counts = count_elements(labels)
    print("label_counts: ", label_counts)
    most_common_label = max(label_counts, key=lambda k: label_counts[k])
    return most_common_label

def count_elements(lst):
    element_counts = {}
    for element in lst:
        if element in element_counts:
            element_counts[element] += 1
        else:
            element_counts[element] = 1
    return element_counts


def find_most_informative_feature(data, label, class_list):
    print("starting finding the most informative feature")
    print("data:",data)
    # Get the feature columns (all columns except the label column)
    feature_list = data.columns[:-1].tolist()
    print("feature_list: ", feature_list)
    max_info_gain = -1
    max_info_feature = None
    print("class_list: ", class_list)
    
    for feature in feature_list:  #for each feature in the dataset
        print("starting calculating feaure gain")
        feature_info_gain = calculate_info_gain(feature, data, label, class_list)
        if max_info_gain < feature_info_gain: #selecting feature name with highest information gain
            max_info_gain = feature_info_gain
            max_info_feature = feature
        print("max info feature: ", max_info_feature)
            
    return max_info_feature

def choose_best_attribute(S, Attributes, Label, class_list, purity_measurement='IG'):
    print("starting choosing best attribute...")
    print("class_list: ", class_list)
    if purity_measurement == 'IG':
        print("starting use purity measurement as Information Gain")
        # Calculate information gain for all attributes and return the one with the maximum IG
        best_attribute = find_most_informative_feature(S, Label, class_list)
    # elif purity_measurement == 'majority':
    #     best_attribute = max(Attributes, key=lambda attr: calculate_me_gain(S, attr, Label))
    # elif purity_measurement == 'gini':
    #     best_attribute = max(Attributes, key=lambda attr: calculate_gini_gain(S, attr, Label))
    # else:
    #     print("Invalid purity measurement input. Only 'IG', 'majority', and 'gini' are supported. Defaulting to Information Gain.")
    #     best_attribute = max(Attributes, key=lambda attr: calculate_information_gain(S, attr, Label))

    return best_attribute


def ID3(S, Attributes, Label, purity_measurement=None):
    # S is a dataframe of the dataset
    # Attributes are the value of the attributes
    # Label are the list of the target variable for the datase
    #Default purity measurement for ID3
    
    if not purity_measurement:
        purity_measurement = 'IG'  # Default purity measurement
    print('purity_measurement:', purity_measurement)

    # Check if leaf mode with the same label
    unique_labels = set(Label)
    class_list = list(unique_labels)
    print("unique_labels: ", unique_labels)
    print("length of unique labels: ", len(unique_labels))
    print("class_list: ", class_list)
    if len(unique_labels) == 1:
        print('unique labels == 1')
        return TreeNode(label=unique_labels.pop())
    # Check if attribute is empty
    elif not Attributes:
        most_common_label = find_most_common_label(Label)
        print("attibute is empty, find most common label: ", most_common_label)
        return TreeNode(label=most_common_label)
    else:
        # Create a Root Node for tree
        root = TreeNode()
        print("starting create root node")
        # Choose the best attribute A to split S
        # Support the input purity measurement of IG, majorty and gini. Here we set default is IG.
        best_attribute = choose_best_attribute(S, Attributes, Label, class_list,purity_measurement)
        print("best attribute: ", best_attribute)
        root.attributes = best_attribute

        # Remove the chosen attribute from the list of attributes
        remaining_attributes = [attr for attr in Attributes if attr != best_attribute]
        print("remaining_attributes: ", remaining_attributes)

        # Split S into subsets based on the values of the best attribute
        attribute_values = S[best_attribute].unique()
        print("choosen attribute_values: ", attribute_values)

        # deal with the remaining attributes for subset Sv, according to A=V
        for value in attribute_values:
            print("value in attribute values:", value)
            print("best_attribute in attribute values:", best_attribute)
            # print(S)
            Sv = S[S[best_attribute] == value]
            remaining_label = Sv.iloc[:,-1].tolist()
            # print("value: {}, Sv {}".format(value, Sv))
            # If Sv is empty, add leaf node with the most common value of label in S
            if Sv.empty:
                most_common_label = find_most_common_label(Label)
                root.children[value] = TreeNode(label=most_common_label)
            else:
                return ID3(Sv, remaining_attributes, remaining_label, purity_measurement)
    return root

def calculate_feature_entropy(feature_value_data, class_list):
    # feature_value_data: Subdataset with feature value data
    # class_list: the unique class list in target variable
    feature_total_count = feature_value_data.shape[0]
    label_list = feature_value_data.iloc[:, -1].tolist()
    print("feature_total_count: ", feature_total_count)
    entropy = 0
    
    for class_label in class_list:
        class_count = label_list.count(class_label)
        print("class_count: ", class_count)
        class_probability = class_count / feature_total_count
        if class_probability > 0:
            class_entropy = -class_probability * math.log2(class_probability)
            entropy += class_entropy

    return entropy

def calculate_target_entropy(labels, class_list):
    #labels: the list of the label values in all data set, eg. df.iloc[:,-1].tolist()
    # class_list: the unique class list in target variable
    
    target_entropy = 0.0
    total_count = len(labels)
    
    for class_label in class_list:
        class_count = labels.count(class_label)
        print("class_label: ", class_label)
        print("class_count: ", class_count)
        class_probability = class_count / total_count
        if class_probability > 0:
            class_entropy = -class_probability * math.log2(class_probability)
            target_entropy += class_entropy
    
    return target_entropy

def calculate_info_gain(feature_name, data, label, class_list):
    print("starting calculating {} information gain".format(feature_name))
    feature_value_list = data[feature_name].unique()
    print("feature_value_list: ",feature_value_list )
    total_row = data.shape[0]
    print("total_row: ", total_row)
    feature_info = 0.0
    
    for feature_value in feature_value_list:
        print("starting feature_value={} in the list".format(feature_value))
        feature_value_data = data[data[feature_name] == feature_value] #filtering rows with that feature_value
        print("feature_value_data: ", feature_value_data)
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calculate_feature_entropy(feature_value_data, class_list) #calculcating entropy for the feature value
        feature_value_probability = feature_value_count/total_row
        feature_info += feature_value_probability * feature_value_entropy #calculating information of the feature value
        
    return calculate_target_entropy(label, class_list) - feature_info #calculating information gain by subtracting
