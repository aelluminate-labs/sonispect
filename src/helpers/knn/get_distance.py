import numpy as np
import operator

def distance(instance1, instance2, k):
    # :: Extract the mean vectors and covariance matrices
    mm1 = instance1[0]  # :: mean vector of instance1
    mm2 = instance2[0]  # :: mean vector of instance2
    cm1 = instance1[1]  # :: covariance matrix of instance1
    cm2 = instance2[1]  # :: covariance matrix of instance2

    # :: Ensure proper dimensions for calculations
    # :: Reshape mean vectors to column vectors (n x 1)
    mm1 = mm1.reshape(-1, 1)
    mm2 = mm2.reshape(-1, 1)

    # :: Ensure covariance matrices are proper matrices
    if cm1.ndim == 0:
        cm1 = np.eye(mm1.shape[0]) * cm1
    if cm2.ndim == 0:
        cm2 = np.eye(mm2.shape[0]) * cm2

    # :: Add small regularization term to ensure matrices are invertible
    epsilon = 1e-6
    cm1 += np.eye(cm1.shape[0]) * epsilon
    cm2 += np.eye(cm2.shape[0]) * epsilon

    try:
        # :: Calculate components of the distance
        # :: 1. Trace term
        term1 = np.trace(np.dot(np.linalg.inv(cm2), cm1))
        
        # :: 2. Mean difference term
        diff = mm2 - mm1
        term2 = np.dot(np.dot(diff.T, np.linalg.inv(cm2)), diff)[0, 0]  # :: Extract scalar value
        
        # :: 3. Log determinant term
        term3 = np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
        
        # :: Combine terms
        dist = 0.5 * (term1 + term2 + term3 - k)
        
        return float(dist)
    except np.linalg.LinAlgError:
        # :: Return a large distance if there's a numerical error
        return float('inf')

def get_distance(training_set, instance, k):
    distances = []
    
    for x in range(len(training_set)):
        # :: Calculate symmetric distance
        dist = distance(training_set[x], instance, k) + distance(instance, training_set[x], k)
        distances.append((training_set[x][2], dist))
    
    # :: Sort by distance and return k nearest neighbors
    distances.sort(key=operator.itemgetter(1))
    neighbors = [distances[x][0] for x in range(min(k, len(distances)))]
    
    return neighbors