import sys
sys.path.append("../Decision Tree")
from DecisionTree import ID3

class BaggedTrees:
    def __init__(self, n_trees):
        self.n_trees = n_trees
        self.trees = []

    def fit(self, data, attributes):
        for _ in range(self.n_trees):
            # 1. Sample with replacement from data
            bootstrap_sample = data.sample(n=len(data), replace=True)
            
            # 2. Train a decision tree on this sample
            tree = ID3(bootstrap_sample, attributes, float('inf'))
            self.trees.append(tree)

    def predict_tree(self, tree, instance):
        node = tree
        while node.children: 
            attribute_name = node.attributes 
            attribute_value = instance[attribute_name]
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
        return node.label

    def predict(self, dataset):
        all_predictions = []

        # For each instance in the dataset
        for _, instance in dataset.iterrows():
            # Predict with each tree and vote
            predictions = [self.predict_tree(tree, instance) for tree in self.trees]
            # Append the majority vote to all_predictions
            all_predictions.append(max(set(predictions), key=predictions.count))

        return all_predictions


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