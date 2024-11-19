def get_accuracy(test_labels, predictions):
    # :: Initialize a counter to keep track of the number of correct predictions
    if len(test_labels) != len(predictions):
        raise ValueError("Number of test labels and predictions must match")
    
    # :: Count the number of correct predictions
    correct = sum(1 for true, pred in zip(test_labels, predictions) if true == pred)

    # :: Calculate and return accuracy
    return correct / len(test_labels) if len(test_labels) > 0 else 0.0