import pickle
import random

# :: Initialize an empty list to store the dataset
dataset = []


def load_dataset(filename, split, tr_set, ts_set):
    # :: Open the file in binary mode
    with open(filename, "rb") as f:
        while True:
            try:
                # :: Load the serialized data form the file adn append to the dataset
                dataset.append(pickle.load(f))
            except EOFError:
                # :: Close the file when the en of the file is reached
                f.close()
                break
    # :: Iterate over each item in the dataset
    for x in range(len(dataset)):
        # Randomly decide whether to add the item to the training set or the test set
        if random.random() < split:
            tr_set.append(dataset[x])
        else:
            ts_set.append(dataset[x])
