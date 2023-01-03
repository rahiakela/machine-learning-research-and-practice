import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        # KNN does nothing but memorizing the data
        self.x_train = X
        self.y_train = y

    def predict(self, X):
        # all heavy lifting is done here
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # compute distance
        distance = [euclidean_distance(x, x_train) for x_train in self.x_train]

        # get the closet k
        k_indices = np.argsort(distance)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # get majority vote
        most_common_labels = Counter(k_nearest_labels).most_common()
        return most_common_labels[0][0]