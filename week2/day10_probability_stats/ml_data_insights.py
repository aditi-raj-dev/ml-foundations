import numpy as np

data = np.array([10, 12, 13, 14, 15])
outlier_data = np.array([10, 12, 13, 14, 500]) # outliers data is destroying mean

print("Normal Data Mean:", np.mean(data))
print("Outlier Data Mean:", np.mean(outlier_data))

print("Normal Data Median:", np.median(data))
print("Outlier Data Median:", np.median(outlier_data))

print("""
Insight:
Outliers destroy mean.
Median survives.
This is why ML preprocessing matters.
""")
