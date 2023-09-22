import pandas as pd
import numpy as np
import math
import TreeNode

class TreeNode: 
    def __init__(self, label=None, attributes=None, children=None):
        self.label = label  # value of the node
        self.attributes = attributes
        self.children = children or []  # dict of child nodes
    
    def __str__(self, level=0):
        prefix = "  " * level
        result = prefix + f"Attribute: {self.attributes}, Label: {self.label}\n"
        for child in self.children:
            result += prefix + f"Child:\n"
            result += child.__str__(level + 1)
        return result
    # def __str__(self):
    #     return str(self.value)

    def add_child(self, child_node):
        self.children.append(child_node)
        
def ID3(S, Attributes, max_depth, purity_measurement=None, root=None):
    if not purity_measurement:
        purity_measurement = 'entropy'  # Default purity measurement
    # Check if leaf mode with the same label
    Label = S.iloc[:,-1].tolist()
    unique_labels = set(Label)
    class_list = list(unique_labels)  
    if len(unique_labels) == 1:
        get_unique_label = next(iter(unique_labels))
        if root is None:
            return TreeNode(label=get_unique_label)
        else:
            root.label = get_unique_label
            return root
    
    # Check if attribute is empty
    elif not Attributes or max_depth==0:
        most_common_label = find_most_common_label(S)
        if root is None:
            return TreeNode(label=most_common_label)
        else:
            root.label = most_common_label
            return root
    
    else:
        if root is None:
            root = TreeNode()
            print("------Root is none, first time creating root node-------")
        print("------starting creating root node-------")
        # Choose the best attribute A to split S
        print("check current root:", root)
        best_attribute = find_best_attribute(S, Attributes, class_list, purity_measurement)
        print("best attribute: ", best_attribute)
        root.attributes = best_attribute 
        
        # Create a new list for remaining attributes
        remaining_attributes = [attr for attr in Attributes if attr != best_attribute]
        print("remaining_attributes: ", remaining_attributes)  
        # Deal with the remaining attributes for subset Sv, according to A=V
        attribute_values = S[best_attribute].unique().tolist()
        
        for value in attribute_values:
            print("value in attribute values:", value)
            child_node = TreeNode()
            child_node.attributes = value  
            root.add_child(child_node) 
            print("Check current root: ", root)

            Sv = S[S[best_attribute] == value]
            
            # If Sv is empty, add leaf node with the most common value of label in S
            if Sv.empty:
                most_common_label = find_most_common_label(Sv)
                child_node.label = most_common_label              
                print("check child node: ", child_node)
            else:
                print("starting create sub tree")
                ID3(Sv, remaining_attributes, max_depth-1, purity_measurement, root=child_node)
    return root


def calculate_entropy(feature_value_data, class_list):
    """Function to calculate entropy"""
    # feature_value_data: Subdataset with feature value data
    # class_list: the unique class list in target variable
    feature_total_count = feature_value_data.shape[0]
    label_list = feature_value_data.iloc[:, -1].tolist()
    entropy = 0
    
    for class_label in class_list:
        class_count = label_list.count(class_label)
        class_probability = class_count / feature_total_count
        if class_probability > 0:
            class_entropy = -class_probability * math.log2(class_probability)
            entropy += class_entropy

    return entropy

def calculate_majority_error(feature_value_data, class_list):
    """Function to calculate majority error"""
    feature_total_count = feature_value_data.shape[0]
    label_list = feature_value_data.iloc[:, -1].tolist()
    majority_error = 0
    
    majority_class_count = max([label_list.count(c) for c in class_list])
    majority_error = 1 - (majority_class_count / feature_total_count)
    
    return majority_error

def calculate_gini_index(feature_value_data, class_list):
    """Function to calculate gini index"""
    feature_total_count = feature_value_data.shape[0]
    label_list = feature_value_data.iloc[:, -1].tolist()
    gini_index = 0
    
    for class_label in class_list:
        class_count = label_list.count(class_label)
        class_probability = class_count / feature_total_count
        gini_index += class_probability * (1 - class_probability)
    
    return gini_index


def calculate_info_gain(feature_name, data, class_list, purity_measurement):
    """Function to calculate information gain for a specfici feature/attribute"""
    # purity_measurement should be one of entropy, majority_error, gini
    feature_value_list = data[feature_name].unique()
    total_row = data.shape[0]
    feature_info = 0.0
    
    for feature_value in feature_value_list:
        feature_value_data = data[data[feature_name] == feature_value] #filtering rows with that feature_value
        feature_value_count = feature_value_data.shape[0]
        if purity_measurement == 'entropy':
            total_entropy = calculate_entropy(data, class_list)
            feature_value_entropy = calculate_entropy(feature_value_data, class_list) #calculcating entropy for the feature value
            feature_value_probability = feature_value_count/total_row
            feature_info += feature_value_probability * feature_value_entropy #calculating information of the feature value
            
        elif purity_measurement == 'majority_error':
            total_entropy = calculate_majority_error(data, class_list)
            feature_value_entropy = calculate_majority_error(feature_value_data, class_list) #calculcating entropy for the feature value
            feature_value_probability = feature_value_count/total_row
            feature_info += feature_value_probability * feature_value_entropy #calculating information of the feature value
        elif purity_measurement == 'gini':
            total_entropy = calculate_gini_index(data, class_list)
            feature_value_entropy = calculate_gini_index(feature_value_data, class_list) #calculcating entropy for the feature value
            feature_value_probability = feature_value_count/total_row
            feature_info += feature_value_probability * feature_value_entropy #calculating information of the feature value
        else: 
            print("No valid purity_measurement has been input. We only support entropy, majority_error and gini. Please further check! ")
            raise ValueError
    return total_entropy - feature_info


def find_best_attribute(data, attributes, class_list, purity_measurement):
    print("starting finding the best feature")
    # Get the feature columns (all columns except the label column)
    print("data count for input df: ", data.shape[0])
    feature_list = data.columns[:-1].tolist()
    print("feature list in find best_attribute", feature_list)
    max_info_gain = -1
    max_info_feature = None

    for attribute in attributes:  #for each feature in the dataset
        feature_info_gain = calculate_info_gain(attribute, data, class_list, purity_measurement)
        print("For {} the information gain is {}".format(attribute, feature_info_gain))
        if max_info_gain < feature_info_gain: #selecting feature name with highest information gain
            max_info_gain = feature_info_gain
            max_info_feature = attribute
            
    return max_info_feature

# Find most common label
def find_most_common_label(df):
    labels = df.iloc[:, -1].tolist()
    label_counts = count_elements(labels)
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