import numpy as np

class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y, epochs=1000, track_loss=False):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        
        # initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        losses = []

        for _ in range(epochs):
            y_pred = np.dot(X, self.weights) + self.bias

            # compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            # update params
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if track_loss:
                loss = np.mean((y_pred - y) ** 2)
                losses.append(loss)

        if track_loss:
            return losses

    def predict(self, X):
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias
