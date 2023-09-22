import pandas as pd

def calculate_error_rate(predictions, true_labels):
    """
    predictions: list of prediction labels using ID3
    true_labels: real labels
    error_rate
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

def predict(tree, test_data):
    """
    tree: ID3 returns
    use this tree to predict EACH label in test data
    """
    while tree.children:
        current_attribute = tree.attributes
        attribute_value = test_data.get(current_attribute)

        matched_child = None
        for child in tree.children:
            if child.attributes == attribute_value:
                matched_child = child
                break

        if matched_child:
            tree = matched_child
        else:
            return tree.label
        
    return tree.label


def make_predictions(tree_root, test_data):
    #Return a list of predictions
    predictions = []
    for index, row in test_data.iterrows():
        prediction = predict(tree_root, row)  
        predictions.append(prediction)
    return predictions

def read_data(file_path, column_names):
    # Read the CSV file with the specified column names
    df = pd.read_csv(file_path, names=column_names)
    # Extract the list of attributes from the DataFrame's columns and drop the last column (label)
    attributes = df.columns.tolist()[:-1]
    return df, attributes