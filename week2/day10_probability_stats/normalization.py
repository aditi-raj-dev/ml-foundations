import numpy as np

data = np.array([10, 50, 80, 100])

print("Original:", data)

# Min Max
min_val = np.min(data)
max_val = np.max(data)
minmax = (data - min_val) / (max_val - min_val)

print("\nMin-Max Normalization:")
print(minmax)

# Z Score
mean = np.mean(data)
std = np.std(data)
zscore = (data - mean) / std

print("\nZ-score Normalization:")
print(zscore)

print("""
Models that NEED normalization:
- Linear Regression
- Logistic Regression
- Neural Networks
- SVM
- KNN
Because they depend on distance or gradient stability.

Models that don't care much:
- Decision Trees
- Random Forest
- XGBoost
Because they split based on rules, not distances.
""")
