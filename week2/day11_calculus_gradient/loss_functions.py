import numpy as np

print("=== LOSS FUNCTIONS ===")

y_true =np.array([2,4,6,8])

# Predictions: one is fairly good , another is very bad
y_pred1 = np.array([2.5, 3.8, 5.5 , 8.2])
y_pred2 = np.array([1,7,2,10])

#---MEAN ABSOLUTE ERROR---
def mae(y_true,y_pred):
      return np.mean(np.abs(y_true-y_pred))

#---MEAN SQUARED ERROR---
def mse(y_true,y_pred):
     return np.mean((y_true-y_pred)**2)

print("\n---MAE---")
print("MAE Prediction 1:", mae(y_true,y_pred1))
print("MAE Prediction 2:", mae(y_true,y_pred2))

print("\n---MSE---")
print("MSE Prediction 1:", mse(y_true,y_pred1))
print("MSE Prediction 2:", mse(y_true,y_pred2))


print("""
Why MSE punishes outliers more?
Because we square the error.
Big mistakes become  MUCH bigger.
This makes the model learn harder when errors are large.
""")

# Loss tells how bad our model is performing. Higher loss = worse predictions.

