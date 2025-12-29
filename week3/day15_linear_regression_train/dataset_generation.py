import numpy as np

np.random.seed(42)

# 1️⃣ Clean Linear Dataset
X_clean = np.linspace(1, 100, 100).reshape(-1, 1)
y_clean = 3 * X_clean + 10

# 2️⃣ Noisy Dataset
noise = np.random.randn(100, 1) * 10
y_noisy = 3 * X_clean + 10 + noise

# 3️⃣ Dataset With Outliers
y_outliers = 3 * X_clean + 10 + noise
y_outliers[::10] += 200   # every 10th point has huge error

def dataset_summary(name, X, y):
    print(f"\n----- {name} DATASET -----")
    print("Shape:", X.shape)
    print("Mean of Y:", np.mean(y))
    print("Variance of Y:", np.var(y))
    print("Range of Y:", np.min(y), "to", np.max(y))
    print("First 5 samples:")
    print("X:", X[:5].flatten())
    print("Y:", y[:5].flatten())

dataset_summary("CLEAN", X_clean, y_clean)
dataset_summary("NOISY", X_clean, y_noisy)
dataset_summary("OUTLIERS", X_clean, y_outliers)

# Export for other files
X = X_clean
y = y_noisy
