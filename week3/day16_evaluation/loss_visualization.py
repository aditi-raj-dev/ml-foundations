import numpy as np
import matplotlib.pyplot as plt
from linear_regression_scratch import LinearRegressionScratch

X = np.array([[1],[2],[3],[4],[5]], dtype=float)
y = np.array([2,4,6,8,10], dtype=float)

model = LinearRegressionScratch(learning_rate=0.01)
losses = model.fit(X, y, epochs=2000, track_loss=True)

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Loss Curve")
plt.show()

