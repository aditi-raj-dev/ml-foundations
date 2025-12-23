import numpy as np

print("=== MATRIX BASICS ===")

# Create a matrix (2 samples, 3 features)
X = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print("Matrix X:")
print(X)
print("Shape of X:", X.shape)

# In ML:
# Rows = samples (data points)
# Columns = features


print("\n=== TRANSPOSE ===")

X_T = X.T
print("Transpose of X:")
print(X_T)
print("Shape of X_T:", X_T.shape)

# Transpose swaps rows and columns
# It is required to make dimensions compatible


print("\n=== MATRIX ADDITION ===")

Y = np.ones((2, 3))
print("Matrix Y:")
print(Y)

Z = X + Y
print("X + Y:")
print(Z)

# Matrix addition works only if shapes are same


print("\n=== MATRIX MULTIPLICATION ===")

# Weight matrix (3 features → 1 output)
W = np.array([
    [1],
    [1],
    [1]
])

print("Weight matrix W:")
print(W)
print("Shape of W:", W.shape)

result = np.dot(X, W)
print("X dot W:")
print(result)
print("Shape of result:", result.shape)

# Each row of X is multiplied with W to produce one output


print("\n=== INTENTIONAL ERROR ===")

try:
    wrong_result = np.dot(W, X)
except ValueError as e:
    print("Error occurred:", e)

# Matrix multiplication rule:
# (a, b) dot (c, d) works ONLY if b == c
