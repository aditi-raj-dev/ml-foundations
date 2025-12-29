from dataset_generation import X, y
from model_linear_regression import LinearRegressionGD

model = LinearRegressionGD(learning_rate=0.0005, epochs=2000)

print("\nTraining Started...\n")
model.fit(X, y)

print("\nValidating on few samples:")
y_pred = model.predict(X[:5])
print("Predicted:", y_pred.flatten())
print("Actual:", y[:5].flatten())
