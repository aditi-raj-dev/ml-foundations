\##1️⃣ Why Train–Test Split Exists



* Model learns patterns from training data



* But we need to check whether it generalizes



* If accuracy is high on train but poor on test → overfitting



* Test set acts like “exam questions never seen before”



* Prevents self-cheating 🙂







2️⃣ What Overfitting REALLY Means (Simple Explanation)



* Model memorizes data instead of learning pattern



* It performs great on training data



* Fails badly on unseen data



* Signs:



&nbsp; - Very low train error



&nbsp; - Very high test error



* Cause:



&nbsp; - Too much learning power



&nbsp; - Too little data



&nbsp; - No regularization







3️⃣ MSE vs MAE (When to Use Which)





\# MSE



* Penalizes large errors heavily (squares them)



* Useful when big mistakes are unacceptable



* Common in regression



\# MAE



* Treats all errors equally



* More robust to outliers





\# Simple summary line:



If outliers present → MAE better

If you want strong punishment → MSE better







4️⃣ Why Loss Visualization Matters





* Training should reduce loss over time



* Helps check:



&nbsp;  - If model is learning



&nbsp;  - If learning rate is good



&nbsp;  - If gradient descent is stable



* If loss curve:



&nbsp; - Goes down smoothly → good



&nbsp; - Explodes upward → LR too high



&nbsp; - Flat → LR too small / bug

&nbsp; 



5️⃣ What Gradient Checking Proves



* We derived formula manually for gradients



* But what if our math/code is wrong?



* Finite difference method:



&nbsp; - Slightly change parameter



&nbsp; - See effect on loss

* Compare:



&nbsp;  - numerical gradient



&nbsp;  - your computed gradient



* If they match → your gradient implementation is correct



* If not → gradient descent training is fake confidence
