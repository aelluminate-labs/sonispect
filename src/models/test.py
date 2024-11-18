from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import os
import pickle
import operator
from collections import defaultdict

# Load the dataset from the pre-trained model
dataset = []


def loadDataset(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break


loadDataset("models/genre_pretrained_model.bat")


# Define the distance function
def distance(instance1, instance2, k):
    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += np.dot(np.dot((mm2 - mm1).T, np.linalg.inv(cm2)), mm2 - mm1)
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance


# Define the function to get neighbors
def getNeighbors(trainingSet, instance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + distance(
            instance, trainingSet[x], k
        )
        distances.append((trainingSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


# Define the function to get the nearest class
def nearestClass(neighbors):
    classVote = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1
    sorter = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]


# Map the genre labels to their corresponding folder names
results = defaultdict(int)
i = 1
for folder in os.listdir("data/raw/sound_clips/"):
    results[i] = folder
    i += 1

# Load the new audio file and extract features
(rate, sig) = wav.read("data/preprocessed/converted/blues/blues.00000_chunk_2.wav")
mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False, nfft=1024)
covariance = np.cov(np.matrix.transpose(mfcc_feat))
mean_matrix = mfcc_feat.mean(0)
feature = (mean_matrix, covariance, 0)

# Predict the genre of the new audio file
pred = nearestClass(getNeighbors(dataset, feature, 5))

# Print the predicted genre
print(f"Predicted Genre: {results[pred]}")
