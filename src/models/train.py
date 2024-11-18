import os
import pickle

import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from src.helpers.load_dataset import load_dataset
from src.models.knn.get_neighbors import get_neighbors
from src.models.knn.get_distance import get_distance
from src.models.knn.get_accuracy import get_accuracy

# :: Extract the features from the dataset and dump them to a file
directory = "data/raw/sound_clips/"

# :: Check if the file exists, if not create it
if not os.path.exists("models/genre_pretrained_model.bat"):
    open("models/genre_pretrained_model.bat", "w").close()

f = open("models/genre_pretrained_model.bat", "wb")
i = 0

for folder in os.listdir(directory):
    print(f"PROCESSING FOLDER: {folder}")
    i += 1

    if i == 11:
        break

    for file in os.listdir(directory + folder):
        file_path = os.path.join(directory + folder, file)
        print(f"PROCESSING FILE: {file_path}")
        if os.path.isfile(file_path) and file_path.endswith(".wav"):
            try:
                (rate, sig) = wav.read(file_path)
                mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
                covariance = np.cov(np.matrix.transpose(mfcc_feat))
                mean_matrix = mfcc_feat.mean(0)
                feature = (mean_matrix, covariance, i)
                pickle.dump(feature, f)
            except Exception as e:
                print(f"PROCESSING ERROR at {file_path}:\n{e}")
                continue
f.close()

# :: Initialize empty lists to store the training and test sets
training_set = []
test_set = []


# :: Load the dataset and split it into training and test sets
load_dataset("models/genre_pretrained_model.bat", 0.66, training_set, test_set)

# :: Print the number of instances in the training and test sets
print("\nTRAINED: " + repr(len(training_set)) + " instances")

# :: Make predictions using the k-NN algorithm
length = len(test_set)
predictions = []

for x in range(length):
    predictions.append(get_neighbors(get_distance(training_set, test_set[x], 5)))

# :: Calculate the accuracy of the predictions
accuracy = get_accuracy(test_set, predictions)

# :: Print the accuracy of the k-NN algorithm
print("ACCURACY: {:.2f}%".format(accuracy * 100))
