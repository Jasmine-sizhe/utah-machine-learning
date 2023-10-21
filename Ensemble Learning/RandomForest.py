# import sys
# sys.path.append("../Decision Tree")
from DecisionTree import ID3
import pandas as pd
import numpy as np


def predict_tree(tree, df_test):
    """
    tree: The tree that has been built from ID3
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

class RandomForest:
    def __init__(self, n_trees, feature_subset_size):
        self.n_trees = n_trees
        self.feature_subset_size = feature_subset_size
        self.trees = []

    def fit(self, data, attributes):
        for _ in range(self.n_trees):
            # Sample with replacement from data
            bootstrap_sample = data.sample(n=len(data), replace=True)
            tree = ID3(S=bootstrap_sample, Attributes=attributes, max_depth = float('inf'),feature_subset_size= self.feature_subset_size, purity_measurement="entropy")
            self.trees.append(tree)

    def predict_single_tree(self, tree_index, dataset):
        if tree_index >= len(self.trees):
            raise ValueError("Tree index out of range.")
        
        return predict_tree(self.trees[tree_index], dataset)

    def predict(self, dataset):
        all_predictions = []

        for _, instance in dataset.iterrows():
            tree_predictions = [predict_tree(tree, pd.DataFrame([instance]))[0] for tree in self.trees]
            all_predictions.append(max(set(tree_predictions), key=tree_predictions.count))

        return all_predictions
