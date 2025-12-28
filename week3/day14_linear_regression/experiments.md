\# DAY 14 — Experiments



\## 1️⃣ Learning Rate Experiment



lr = 0.0001

Very slow. Loss decreases but training takes long.



lr = 0.01

Stable and fast. Best performance.



lr = 1.0

Explodes. Loss becomes huge. Divergence.



---



\## 2️⃣ Initial Weights Experiment

Starting m, b very positive → still converges but slower  

Starting m, b very negative → converges but oscillates slightly first



Conclusion:

Gradient descent works regardless of start, but affects speed.



---



\## 3️⃣ Failure Case

Insane learning rate (lr = 10)

Loss → NaN

Model completely diverges

Training FAILS



This proves:

Learning rate controls success of training.



