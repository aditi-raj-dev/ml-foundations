\# DAY 14 — LINEAR REGRESSION FROM SCRATCH



\## 1️⃣ What is Linear Regression Actually Doing?

Linear Regression tries to learn a straight line that best fits the data.

It learns:

y = mX + c

where:

m = weight (slope)

c = bias (intercept)



It predicts output based on relationship between input and output.

It finds the best line by minimizing error.



---



\## 2️⃣ Why is Linear Regression Used in Industry?

✔ Simple and interpretable  

✔ Very fast  

✔ Works great when relationship is linear  

✔ Used in finance, economics, pricing, forecasting, etc.



---



\## 3️⃣ Mean Squared Error — What It Penalizes?

MSE = mean((y\_true − y\_pred)²)



It penalizes:

• Large mistakes heavily (because squared)

• Outliers strongly

• Wrong predictions more aggressively



Smaller MSE = better model.



---



\## 4️⃣ Gradient Descent Intuition (Not Full Proof)

We don't brute-force search best m and c.

Instead:

1\. Start with random m, c

2\. Look at slope of loss curve (gradient)

3\. Move opposite direction of slope

4\. Repeat until loss stops reducing



Gradient tells:

“Which direction increases error?”

So we go opposite direction to reduce it.



---



\## 5️⃣ Why Scaling Matters?

If features are very large or vary a lot:

• gradient becomes unstable

• training becomes slow

• model may diverge



Scaling keeps values small and stable.



---



\## 6️⃣ When Linear Regression Fails?

❌ Data is not linear  

❌ Extreme outliers  

❌ Features are highly correlated  

❌ Learning rate too high  

❌ Noisy nonlinear data  



Then advanced models are needed.



