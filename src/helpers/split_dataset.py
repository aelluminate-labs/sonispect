from sklearn.model_selection import train_test_split


def split_dataset(features, labels, test_size=0.33, random_state=42):
    """
    Split the dataset into training and test sets.

    :param features: Features of the dataset.
    :param labels: Labels of the dataset.
    :param test_size: Proportion of the dataset to include in the test split.
    :param random_state: Random state for reproducibility.
    :return: Tuple containing training and test sets.
    """
    return train_test_split(features, labels, test_size=test_size, random_state=random_state)