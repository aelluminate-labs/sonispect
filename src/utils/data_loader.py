import pickle
import numpy as np
from typing import List, Tuple, Optional

class DataLoader:
    @staticmethod
    def load_dataset(filename: str, split: Optional[float] = None, model_type: str = "knn") -> Tuple[List, List]:
        # :: Initialize dataset
        dataset = []

        # :: Load raw data from the pickle file
        with open(filename, 'rb') as f:
            while True:
                try:
                    if model_type == "nn":
                        feature, label = pickle.load(f)
                        dataset.append((feature, label))
                    else:
                        mean_matrix, covariance, label = pickle.load(f)
                        dataset.append((mean_matrix, covariance, label))
                except EOFError:
                    break

        # :: If no split is provided, return the whole dataset
        if split is None:
            return dataset, []

        # :: Split data into training and test sets
        training_set = []
        test_set = []

        for data in dataset:
            if np.random.random() < split:
                training_set.append(data)
            else:
                test_set.append(data)

        return training_set, test_set
