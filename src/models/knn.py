import numpy as np
import operator
from typing import List, Tuple, Any

class KNNClassifier:
    def __init__(self, k_neighbors: int = 5):
        self.k = k_neighbors
        self.training_data = None

    def _distance(self, instance1: Tuple[np.ndarray, np.ndarray], 
                instance2: Tuple[np.ndarray, np.ndarray]) -> float:
        # :: Calculate distance between two instances.
        mm1, cm1 = instance1
        mm2, cm2 = instance2
        
        distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
        distance += np.dot(np.dot((mm2-mm1).transpose(), np.linalg.inv(cm2)), mm2-mm1)
        distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
        distance -= self.k
        return distance

    def _get_neighbors(self, instance: Tuple) -> List:
        # :: Find k nearest neighbors for an instance.
        distances = []
        for x in range(len(self.training_data)):
            dist = self._distance((self.training_data[x][0], self.training_data[x][1]), 
                                (instance[0], instance[1])) + \
                   self._distance((instance[0], instance[1]), 
                                (self.training_data[x][0], self.training_data[x][1]))
            distances.append((self.training_data[x][2], dist))
        
        distances.sort(key=operator.itemgetter(1))
        return [distances[x][0] for x in range(self.k)]

    def fit(self, training_data: List[Tuple]):
        # :: Train the classifier.
        
        self.training_data = training_data

    def predict(self, instances: List[Tuple]) -> List:
        # :: Predict classes for multiple instances.
        return [self.predict_single(instance) for instance in instances]

    def predict_single(self, instance: Tuple) -> Any:
        # :: Predict class for a single instance.
        neighbors = self._get_neighbors(instance)
        return self._nearest_class(neighbors)

    def _nearest_class(self, neighbors: List) -> Any:
        # :: Determine most common class among neighbors.
        class_vote = {}
        for response in neighbors:
            class_vote[response] = class_vote.get(response, 0) + 1
        
        sorted_votes = sorted(class_vote.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_votes[0][0]