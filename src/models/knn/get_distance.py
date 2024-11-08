import numpy as np
import operator


def distance(instance1, instance2, k):
    # :: Initialize the distance to 0
    dist = 0

    # :: Extract the mean vectors from the instances
    mm1 = instance1[0]
    mm2 = instance2[0]

    # :: Extract the covariance matrices from the instances
    cm1 = instance1[1]
    cm2 = instance2[1]

    # :: Calculate the Mahalanobis distance between the two instances
    # :: Compute the trace of the product of the inverse of cm2 and cm1
    dist += np.trace(np.dot(np.linalg.inv(cm2), cm1))

    # :: Compute the Mahalanobis distance between the mean vectors
    # :: NOTE: The transpose method is corrected to .T
    dist += np.dot(np.dot((mm2 - mm1).T, np.linalg.inv(cm2)), mm2 - mm1)

    # :: Add the difference in the log determinants of the covariance matrices
    dist += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))

    # :: Subtract the dimensionality of the data (k)
    dist -= k

    # :: Return the computed distance
    return dist


def get_distance(training_set, instance, k):
    # :: Initialize an empty list to store the distances
    distances = []

    # :: Iterate over each element in the training set
    for x in range(len(training_set)):
        # :: Calculate the distance between the current training instance and the given instance
        # :: NOTE: The distance function is called twice with swapped arguments.
        dist = distance(training_set[x], instance, k) + distance(
            instance, training_set[x], k
        )
        # :: Append the training instance label and the calculated distance to the distances list
        distances.append((training_set[x][2], dist))

    # :: Sort the distances list by the second element in the tuple (the distance)
    distances.sort(key=operator.itemgetter(1))

    # :: Initialize an empty list to the k-nearest neighbors
    neighbors = []

    # :: Iterate over the first k elements in the sorted distances list
    for x in range(k):
        # :: Append the label of the nearest neighbor to the neighbors list
        neighbors.append(distances[x][0])

    # :: Return the list of k-nearest neighbors
    return neighbors
