import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, lr=0.001, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            # predict result
            logits = np.dot(X, self.weights) + self.bias
            preds = sigmoid(logits)

            # updating parameters
            dw = (1 / n_samples) * np.dot(X.T, (preds - y))
            db = (1 / n_samples) * np.sum(preds - y)
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        logits = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(logits)
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred

    def accuracy(self, y_pred, y_test):
        return np.sum(y_pred==y_test) / len(y_test)
