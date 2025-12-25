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

## Why probability stabilizes ML?
# Because ML assumes data follows a distribution.
# When distributions are known , models become predictable and stable instead of guessing randomly.

## Why randomness matters in datasets ?
# Real - world data always has noise . Randomness helps models generalize instead of memorizing.
# Without randomness , models would overfit and fail in real - world conditions .

