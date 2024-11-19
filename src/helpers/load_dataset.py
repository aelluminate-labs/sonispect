import pickle
import numpy as np


def load_dataset(input_file):
    """
    Load the dataset from the specified file.

    :param input_file: Path to the file containing the features.
    :return: Tuple containing features and labels.
    """
    features = []
    labels = []

    with open(input_file, "rb") as f:
        while True:
            try:
                feature = pickle.load(f)
                features.append(feature[0])
                labels.append(feature[2])
            except EOFError:
                break

    features = np.array(features)
    labels = np.array(labels) - 1
    return features, labels