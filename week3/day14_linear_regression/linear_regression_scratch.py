import numpy as np

class LinearRegressionScratch:
    def __init__(self, learning_rate=0.001):
        self.lr = learning_rate
        self.m = None
        self.b = None

    def predict(self, X):
        return X * self.m + self.b

    def fit(self, X, y, epochs=1000):
        n = len(X)

        # initialize parameters
        self.m = np.random.randn()
        self.b = np.random.randn()

        for i in range(epochs):
            y_pred = self.predict(X)

            # Mean Squared Error
            loss = np.mean((y - y_pred) ** 2)

            # Gradients (derived manually)
            dm = (-2/n) * np.sum(X * (y - y_pred))
            db = (-2/n) * np.sum(y - y_pred)

            # Update rule
            self.m -= self.lr * dm
            self.b -= self.lr * db

            if i % 100 == 0:
                print(f"Epoch {i} | Loss = {loss:.4f} | m={self.m:.3f} | b={self.b:.3f}")

        print("\nTraining Completed.")
        print(f"Final m: {self.m}")
        print(f"Final b: {self.b}")
