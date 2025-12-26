import numpy as np

# True values
y_true = np.array([3, 5, 7, 9])

# Predictions from a model
y_pred = np.array([2.5, 4.2, 6.5, 10])

print("Y True:", y_true)
print("Y Pred:", y_pred)

print("\n=== STEP 1: Find Errors ===")
errors = y_true - y_pred
print("Errors (y_true - y_pred):", errors)

print("\n=== STEP 2: Square Errors ===")
squared_errors = errors ** 2
print("Squared Errors:", squared_errors)

print("\n=== STEP 3: Mean of Squared Errors ===")
mse_manual = np.sum(squared_errors) / len(y_true)
print("MSE (Manual Calculation):", mse_manual)

print("\n=== STEP 4: Verify with NumPy ===")
mse_numpy = np.mean((y_true - y_pred) ** 2)
print("MSE (NumPy):", mse_numpy)

print("\nBoth values must match → then your understanding is correct.")
