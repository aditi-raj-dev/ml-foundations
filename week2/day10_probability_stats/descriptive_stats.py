import numpy as np

data = np.array([10, 20, 30, 40, 100])

print("Data:", data)

# Manual Mean
mean_manual = np.sum(data) / len(data)
mean_np = np.mean(data)

print("\nMean")
print("Manual:", mean_manual)
print("NumPy:", mean_np)

# Manual Median
sorted_data = np.sort(data)
mid = len(data) // 2
median_manual = sorted_data[mid]
median_np = np.median(data)

print("\nMedian")
print("Manual:", median_manual)
print("NumPy:", median_np)

# Manual Variance
variance_manual = np.sum((data - mean_manual) ** 2) / len(data)
variance_np = np.var(data)

print("\nVariance")
print("Manual:", variance_manual)
print("NumPy:", variance_np)

# Standard Deviation
std_manual = np.sqrt(variance_manual) # square root of variance is standard  deviation
std_np = np.std(data)

print("\nStandard Deviation")
print("Manual:", std_manual)
print("NumPy:", std_np)
