import numpy as np

print("=== LINEAR REGRESSION: MATRIX FORM ===")

# Dataset matrix (X)
# Each row = one data sample
# Each column = one feature
X = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

# Weight vector (W)
# One weight per feature
W = np.array([
    [0.5],
    [1.0]
])

# Bias (b)
b = 2

print("\n--- SHAPES ---")
print("X shape:", X.shape)
print("W shape:", W.shape)

print("\n--- LINEAR EQUATION ---")
print("y = XW + b")

# Matrix multiplication
XW = np.dot(X, W)

print("\nXW (before bias):")
print(XW)
print("XW shape:", XW.shape)

# Add bias
y = XW + b

print("\nFinal output y:")
print(y)
print("y shape:", y.shape)
