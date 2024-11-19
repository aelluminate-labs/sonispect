import operator


def get_neighbors(neighbors):
    # :: If there are no neighbors, return None
    if not neighbors:
        return None

    # :: Count votes for each class
    class_votes = {}
    for label in neighbors:
        class_votes[label] = class_votes.get(label, 0) + 1

    # :: Find the class with maximum votes; In case of tie, returns the first one encountered
    return max(class_votes.items(), key=lambda x: x[1])[0]
