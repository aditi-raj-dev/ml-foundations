import numpy as np
from dataset import X, y
from linear_regression_scratch import LinearRegressionScratch

model = LinearRegressionScratch(learning_rate=0.0005)

print("Training Started...\n")
model.fit(X, y, epochs=2000)

print("\nTraining Finished Successfully.")
