import numpy as np
from src.helpers.extract_features import extract_features
from src.helpers.load_dataset import load_dataset
from src.helpers.normalize_features import normalize_features
from src.helpers.split_dataset import split_dataset
from src.helpers.knn.get_neighbors import get_neighbors
from src.helpers.knn.get_distance import get_distance
from src.helpers.knn.get_accuracy import get_accuracy

# :: Define paths
directory = "data/raw/sound_clips/"
output_file = "models/genre_pretrained_model.bat"

# :: Extract features
instance_count = extract_features(directory, output_file, max_folders=10)

# :: Print Model Name
print("\n:: KNN Model ::")
print(f"PROCESSED: {instance_count} instances")

# :: Load dataset
features, labels = load_dataset(output_file)

# :: Normalize features
features = normalize_features(features)

# :: Split dataset
X_train, X_test, y_train, y_test = split_dataset(features, labels)

# :: Print the number of instances in the training and test sets
print("TRAINED: " + repr(len(X_train)) + " instances")
print("TESTED: " + repr(len(X_test)) + " instances")

# :: Format features for k-NN algorithm
X_train_formatted = [(x, np.cov(x.T), y) for x, y in zip(X_train, y_train)]
X_test_formatted = [(x, np.cov(x.T), y) for x, y in zip(X_test, y_test)]

# :: Make predictions using the k-NN algorithm
length = len(X_test_formatted)
predictions = []

for x in range(length):
    predictions.append(get_neighbors(get_distance(X_train_formatted, X_test_formatted[x], 5)))

# :: Calculate the accuracy of the predictions
accuracy = get_accuracy(y_test, predictions)

# :: Print the accuracy of the k-NN algorithm
print("ACCURACY: {:.2f}%".format(accuracy * 100))