def get_accuracy(test_set, predictions):
    # :: Initialize a counter to keep track of correct predictions
    correct = 0

    # :: Iterate over each element in the test set
    for x in range(len(test_set)):
        # :: Check if the actual class label (last element of the test set instance) matches the predicted class label
        if test_set[x][-1] == predictions[x]:
            # :: If the match, increment the correct counter
            correct += 1

    # :: Calculate the accuracy as the ratio of correct predictions to the total number of test instances
    # :: Multiply by 1.0 to ensure the result is a floating-point number
    return 1.0 * correct / len(test_set)
