import numpy as np
from dataset import X, y

def train_test_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)

    n_samples = X.shape[0]

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    split_point = int(n_samples * (1 - test_size))

    X_train = X[:split_point]
    X_test = X[split_point:]

    y_train = y[:split_point]
    y_test = y[split_point:]

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y)

print("----- SHAPE REPORT -----")
print("Before Split:", X.shape, y.shape)
print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)
