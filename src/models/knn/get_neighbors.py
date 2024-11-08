import operator


def get_neighbors(neighbors):
    # :: Initialize an empty dictionary to store the votes for each class
    class_votes = {}

    # :: Iterate over each neighbor in the list of neighbors
    for x in range(len(neighbors)):
        # :: Extract the class label from the current neighbor
        response = neighbors[x]
        # :: Check if the class label is already in the dictionary
        if response in class_votes:
            # :: If it is, increment the vote count for that class
            class_votes[response] += 1
        else:
            # :: If it is not, add the class label to the dictionary with a vote count of 1
            class_votes[response] = 1

    # :: Sort the class votes dictionary by the vote counts in descending order
    # :: NOTE: The `operator.itemgetter(1)` method is used to sort by the second element in the tuple
    sorter = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)

    # :: Return the class label with the most votes
    return sorter[0][0]
