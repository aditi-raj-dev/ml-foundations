import numpy as np

print("=== LINEAR REGRESSION: MATRIX FORM ===")

# Dataset matrix (X)
# Each row = one data sample
# Each column = one feature
X_data = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

# Weight vector (W)
# One weight per feature
weights = np.array([
    [0.5],
    [1.0]
])

# Bias (b)
b = 2

print("\n--- SHAPES ---")
print("X shape:", X_data.shape)
print("W shape:", weights.shape)

print("\n--- LINEAR EQUATION ---")
print("y = XW + b")

assert X_data.shape[1] == weights.shape[0], "Shape mismatch: Cannot multiply X and W"


# Matrix multiplication
XW = np.dot(X_data, weights)

print("\nXW (before bias):")
print(XW)
print("XW shape:", XW.shape)

# Add bias
predictions = XW + b

print("\nFinal output y:")
print(predictions)
print("y shape:", predictions.shape)
