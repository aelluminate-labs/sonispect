from typing import List

class Metrics:
    @staticmethod
    def accuracy(test_set: List, predictions: List) -> float:
        # :: Calculate accuracy of predictions.
        correct = sum(1 for x in range(len(test_set)) if test_set[x][-1] == predictions[x])
        return correct / len(test_set)