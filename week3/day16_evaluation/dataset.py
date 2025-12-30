import numpy as np

np.random.seed(42)

# Create feature X
X = np.linspace(1, 100, 100).reshape(-1, 1)

# True relation y = 2x + 5 + noise
true_m = 2
true_c = 5
noise = np.random.randn(100, 1) * 5

y = true_m * X + true_c + noise

print("----- DATASET SUMMARY -----")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("First 5 X values:\n", X[:5])
print("First 5 y values:\n", y[:5])

print("""
Meaning:
X = input feature (example: house size, years experience)
y = output generated using linear relationship + noise
Noise makes dataset realistic, forcing model to learn trend, not memorize.
""")
