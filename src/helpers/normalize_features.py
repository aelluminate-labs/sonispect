from sklearn.preprocessing import StandardScaler


def normalize_features(features):
    """
    Normalize the features using StandardScaler.

    :param features: Features to be normalized.
    :return: Normalized features.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(features)