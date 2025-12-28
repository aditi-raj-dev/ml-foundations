import numpy as np
from dataset import X, y
from linear_regression_scratch import LinearRegressionScratch

model = LinearRegressionScratch(learning_rate=0.0005)
model.fit(X, y, epochs=1500)

y_pred = model.predict(X)

# Final Loss
mse = np.mean((y - y_pred)**2)

# R2 Score
ss_total = np.sum((y - np.mean(y))**2)
ss_res = np.sum((y - y_pred)**2)
r2 = 1 - (ss_res / ss_total)

print("\n----- EVALUATION -----")
print("Final MSE:", mse)
print("R² Score:", r2)

if r2 > 0.9:
    print("Model learned VERY WELL 👍")
elif r2 > 0.7:
    print("Model is decent 👍")
else:
    print("Model underfitting ❌ Needs tuning")
