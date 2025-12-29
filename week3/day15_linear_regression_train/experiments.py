from dataset_generation import X, y
from model_linear_regression import LinearRegressionGD

print("\n===== LEARNING RATE EXPERIMENTS =====")

for lr in [0.0001, 0.01, 1.0]:
    print(f"\nLearning Rate = {lr}")
    model = LinearRegressionGD(learning_rate=lr, epochs=1000)
    model.fit(X, y)
    print("--------------------------------------")


print("\n===== INITIAL WEIGHT EXPERIMENTS =====")

model1 = LinearRegressionGD(learning_rate=0.001, epochs=1500)
model1.w = 100
model1.b = -100
model1.fit(X, y)

model2 = LinearRegressionGD(learning_rate=0.001, epochs=1500)
model2.w = -200
model2.b = 200
model2.fit(X, y)


print("\n===== EXTREME NOISE FAILURE CASE =====")

import numpy as np
noise = np.random.randn(100, 1) * 100
y_bad = 3 * X + 10 + noise

model_bad = LinearRegressionGD(learning_rate=0.001, epochs=1500)
model_bad.fit(X, y_bad)
print("Observation: Loss remains high because data is extremely noisy")
