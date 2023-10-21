import sys
sys.path.append("../Decision Tree")
from DecisionTree import ID3
import pandas as pd
import matplotlib.pyplot as plt

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

def run_bagged_trees_experiment(train_data, test_data, attributes, n_iterations=10, n_trees=20, sample_size=1000):
    
    def sample_data(train_data, sample_size=1000):
        """Sample data uniformly without replacement."""
        return train_data.sample(n=sample_size, replace=False)
    
    bagged_predictors = []
    single_trees = []

    training_errors = []
    testing_errors = []

    for _ in range(n_iterations):
        sampled_data = sample_data(train_data, sample_size)

        bagged_model = BaggedTrees(n_trees=n_trees)
        bagged_model.fit(sampled_data, attributes)

        train_predictions = bagged_model.predict(sampled_data)
        true_labels_train = sampled_data.iloc[:, -1].tolist()
        error_rate_train = calculate_error_rate(train_predictions, true_labels_train)
        training_errors.append(error_rate_train)
    
        test_predictions = bagged_model.predict(test_data)
        true_labels_test = test_data.iloc[:, -1].tolist()
        error_rate_test = calculate_error_rate(test_predictions, true_labels_test)
        testing_errors.append(error_rate_test)

        bagged_predictors.append(bagged_model)
        single_trees.append(bagged_model.trees[0])

    all_single_tree_predictions = []
    for test_example in test_data.iterrows():
        single_tree_predictions = [predict(tree, pd.DataFrame([test_example[1]])) for tree in single_trees]
        all_single_tree_predictions.append(single_tree_predictions)

    binary_predictions = [[1 if pred[0] == 'yes' else -1 for pred in predictions] for predictions in all_single_tree_predictions]
    true_labels_binary = [1 if label == 'yes' else -1 for label in test_data.iloc[:, -1].tolist()]

    biases = []
    variances = []
    for i, test_example_predictions in enumerate(binary_predictions):
        avg_prediction = np.mean(test_example_predictions)
        bias = (true_labels_binary[i] - avg_prediction) ** 2
        biases.append(bias)
        variance = np.var(test_example_predictions)
        variances.append(variance)

    avg_bias = np.mean(biases)
    avg_variance = np.mean(variances)

    print("Average Bias:", avg_bias)
    print("Average Variance:", avg_variance)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_iterations + 1), training_errors, label='Training Error', marker='o')
    plt.plot(range(1, n_iterations + 1), testing_errors, label='Testing Error', marker='x')
    plt.xlabel('Iterations')
    plt.ylabel('Error Rate')
    plt.title('Training and Testing Errors vs. Iterations')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, n_iterations + 1))
    plt.show()