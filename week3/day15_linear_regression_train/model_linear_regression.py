import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.001, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None

    def predict(self, X):
        return X * self.w + self.b

    def compute_loss(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)

    def step(self, X, y):
        n = len(X)
        y_pred = self.predict(X)

        dw = (-2/n) * np.sum(X * (y - y_pred))
        db = (-2/n) * np.sum(y - y_pred)

        self.w -= self.lr * dw
        self.b -= self.lr * db

    def fit(self, X, y):
        self.w = np.random.randn()
        self.b = np.random.randn()

        print(f"Initial Weight = {self.w:.4f}, Bias = {self.b:.4f}")
        print(f"Initial Loss = {self.compute_loss(X, y):.4f}")

        for epoch in range(self.epochs):
            self.step(X, y)

            if epoch % 200 == 0:
                loss = self.compute_loss(X, y)
                print(f"Epoch {epoch} | Loss = {loss:.4f}")

        print("\nTraining Finished")
        print(f"Final Weight = {self.w:.4f}, Bias = {self.b:.4f}")
        print(f"Final Loss = {self.compute_loss(X, y):.4f}")
