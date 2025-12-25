# Day 10 — Probability \& Statistics for ML



## What Probability Means in ML

Probability in ML is not about coins and dice.

It is about \*\*how uncertain data and predictions are\*\*.

Whenever a model predicts something, it is never fully sure.

So probability helps ML handle uncertainty instead of guessing blindly.



In ML:

- Data is noisy

- Real world behaves randomly

- We cannot trust a single value

So probability helps understand patterns in uncertainty.



---



## Mean vs Median

Mean = average value

Median = middle value



Mean is good when data is clean.

But mean lies when there are outliers.



Example:

Marks = 20, 21, 22, 23, 100

Mean becomes large because of 100

Median stays realistic.



So:

\- Use Mean when data is stable

\- Use Median when there are outliers



---



\## Variance \& Standard Deviation

Mean tells “center”.

Variance \& Std tell \*\*how spread the data is\*\*.



High variance = data is scattered and unstable

Low variance = data is consistent



Std dev is important in ML because:

\- If data is highly spread, models struggle to learn

\- If data is stable, learning is easier



---


\## Gaussian Distribution

Most real world data tends to form a bell curve.

This happens because many small random effects combine naturally.



ML loves Gaussian because:

\- Gradients behave smoothly

\- Math becomes stable

\- Many algorithms assume normality



---



\## Why Normalization Matters

Different features may have different ranges.



Example:

Height: 160 → 190

Salary: 10,000 → 2,00,000



If not normalized:

Model thinks salary is more important only because numbers are bigger.



Normalization:

\- Brings data to same scale

\- Faster training

\- More stability

\- Better accuracy




Mean = central tendency
Variance = how spread the data is
Std dev= natural unit of spread
Gaussian= appears because many small effects combine (central limit theorem)
Normalization= keeps ML training stable




