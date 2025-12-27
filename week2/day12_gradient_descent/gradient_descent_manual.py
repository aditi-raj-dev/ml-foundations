import numpy as np

print("=== MANUAL GRADIENT DESCENT ===")

# Training Data
X = np.array([1,2,3,4], dtype=float)
y = np.array([3,5,7,9], dtype=float)

# Initialize parameters
m = np.random.randn()   # slope
c = np.random.randn()   # bias

lr = 0.01   # learning rate
epochs = 50

print("Initial m:", m)
print("Initial c:", c)

def predict(X, m, c):
    return m * X + c

def mse(y_true, y_pred):
    error = y_true - y_pred
    squared_error = error ** 2
    return np.mean(squared_error)

def compute_gradients(X, y, y_pred):
    n = len(X)
    
    dm = (-2/n) * np.sum(X * (y - y_pred))   # gradient wrt m
    dc = (-2/n) * np.sum(y - y_pred)         # gradient wrt c
    
    return dm, dc

for i in range(epochs):

    # forward pass
    y_pred = predict(X, m, c)

    # loss
    loss = mse(y, y_pred)

    # gradients
    dm, dc = compute_gradients(X, y, y_pred)

    # update
    m = m - lr * dm
    c = c - lr * dc

    if (i+1) % 5 == 0:
        print(f"Epoch {i+1}: Loss={loss:.4f}, m={m:.4f}, c={c:.4f}")

# Grdaient shows direction of steepest increase of loss.
# We move in opposite direction (-gradient) to reduce loss



