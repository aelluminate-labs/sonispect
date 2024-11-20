import pickle
import random
from typing import List, Tuple

class DataLoader:
    @staticmethod
    def load_dataset(filename: str, split: float = None) -> Tuple[List, List]:
        # :: Load dataset and optionally split into train/test sets.
        dataset = []
        with open(filename, 'rb') as f:
            while True:
                try:
                    dataset.append(pickle.load(f))
                except EOFError:
                    break

        if split is None:
            return dataset, []

        training_set = []
        test_set = []
        
        for item in dataset:
            if random.random() < split:
                training_set.append(item)
            else:
                test_set.append(item)
                
        return training_set, test_set