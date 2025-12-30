import numpy as np
from dataset import X, y
np.random.seed(42)


# -----------------------------
# MODEL (with compute_loss)
# -----------------------------
class LinearRegressionScratch:
    def __init__(self, learning_rate=0.001):
        self.lr = learning_rate
        self.m = None
        self.b = None

    def predict(self, X):
        return X * self.m + self.b

    def compute_loss(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)

    def fit(self, X, y, epochs=1000):
        n = len(X)
        self.m = np.random.randn()
        self.b = np.random.randn()

        for i in range(epochs):
            y_pred = self.predict(X)

            dm = (-2/n) * np.sum(X * (y - y_pred))
            db = (-2/n) * np.sum(y - y_pred)

            self.m -= self.lr * dm
            self.b -= self.lr * db


# -----------------------------
# TRAIN MODEL FIRST
# -----------------------------
model = LinearRegressionScratch(learning_rate=0.0005)
model.fit(X, y, epochs=500)

print("Trained Parameters:")
print("m =", model.m)
print("b =", model.b)


# -----------------------------
# GRADIENT VERIFICATION
# Using Finite Difference
# -----------------------------

epsilon = 1e-5

# Store current values
m_original = model.m
b_original = model.b


# ----- NUMERICAL GRADIENT wrt m -----
model.m = m_original + epsilon
loss1 = model.compute_loss(X, y)

model.m = m_original - epsilon
loss2 = model.compute_loss(X, y)

numerical_grad_m = (loss1 - loss2) / (2 * epsilon)


# ----- ANALYTICAL GRADIENT wrt m -----
y_pred = model.predict(X)
n = len(X)
analytical_grad_m = (-2/n) * np.sum(X * (y - y_pred))


# Reset m
model.m = m_original


# ----- NUMERICAL GRADIENT wrt b -----
model.b = b_original + epsilon
loss1 = model.compute_loss(X, y)

model.b = b_original - epsilon
loss2 = model.compute_loss(X, y)

numerical_grad_b = (loss1 - loss2) / (2 * epsilon)


# ----- ANALYTICAL GRADIENT wrt b -----
analytical_grad_b = (-2/n) * np.sum(y - y_pred)


# Reset b
model.b = b_original


# -----------------------------
# RESULTS
# -----------------------------
print("\n----- GRADIENT VERIFICATION -----")
print(f"Gradient wrt m -> Analytical: {analytical_grad_m:.6f} | Numerical: {numerical_grad_m:.6f}")
print(f"Gradient wrt b -> Analytical: {analytical_grad_b:.6f} | Numerical: {numerical_grad_b:.6f}")

print("\nDifference:")
print("m difference =", abs(analytical_grad_m - numerical_grad_m))
print("b difference =", abs(analytical_grad_b - numerical_grad_b))

print("\nIf differences are VERY SMALL → Gradient is CORRECT 🎯")

