import numpy as np
import time

# -------------------------------------------------------
# Function: Mean Squared Error
# -------------------------------------------------------
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# -------------------------------------------------------
# Gradient Descent Function
# -------------------------------------------------------
def gradient_descent(X, y, learning_rate, epochs):
    m = 0.0   # slope
    c = 0.0   # intercept
    n = len(X)

    losses = []

    for i in range(epochs):
        y_pred = m * X + c

        loss = mse(y, y_pred)
        losses.append(loss)

        # GRADIENTS
        dm = (-2 / n) * np.sum(X * (y - y_pred))
        dc = (-2 / n) * np.sum(y - y_pred)

        # UPDATE RULE
        # Move opposite to gradient direction (DOWNHILL)
        m = m - learning_rate * dm
        c = c - learning_rate * dc

        # Print few logs for understanding
        if i % 10 == 0 or i == epochs - 1:
            print(f"Epoch {i+1}: Loss = {loss:.6f}, m = {m:.4f}, c = {c:.4f}")

    return m, c, losses[-1]

# -------------------------------------------------------
# DATASET (NO NOISE)
# -------------------------------------------------------
X = np.linspace(0, 10, 100)
y = 2 * X + 1   # Perfect linear relationship

print("\n================ EXPERIMENT 1: Learning Rates ================\n")

learning_rates = [0.0001, 0.01, 0.1]

for lr in learning_rates:
    print(f"\n--- Learning Rate = {lr} ---")
    m, c, final_loss = gradient_descent(X, y, learning_rate=lr, epochs=100)

    print(f"Final m = {m:.4f}, Final c = {c:.4f}, Final Loss = {final_loss:.6f}")

    # EXPLANATION IN COMMENTS:
    # lr = 0.0001 -> Very slow learning, converges slowly
    # lr = 0.01   -> Best balance, smooth convergence
    # lr = 0.1    -> Too aggressive, may oscillate or explode


print("\n================ EXPERIMENT 2: Iteration Counts ================\n")

for epochs in [10, 50, 200]:
    print(f"\n--- Epochs = {epochs} ---")
    m, c, final_loss = gradient_descent(X, y, learning_rate=0.01, epochs=epochs)
    print(f"Final m = {m:.4f}, Final c = {c:.4f}, Final Loss = {final_loss:.6f}")

    # THINKING / UNDERSTANDING (written as comments, not chat):
    # 10 epochs  -> Model barely learns
    # 50 epochs  -> Better learning
    # 200 epochs -> Very close to true values (2 and 1)
    # More epochs = more refinement, BUT after some point improvement becomes tiny


print("\n================ EXPERIMENT 3: Noise in Data ================\n")

np.random.seed(42)
noise = np.random.normal(0, 1, 100)

y_noise = 2 * X + 1 + noise   # REALISTIC DATA (not perfect now)

m, c, final_loss = gradient_descent(X, y_noise, learning_rate=0.01, epochs=200)

print(f"\nFinal with Noise → m = {m:.4f}, c = {c:.4f}, Loss = {final_loss:.6f}")

# EXPLANATION:
# Because of noise:
# 1) Loss will NEVER become zero
# 2) m and c will NOT become exactly 2 and 1
# 3) Still, gradient descent finds the BEST POSSIBLE FIT LINE
# This is EXACTLY how real machine learning works
# Real world data always has noise, so models aim for BEST APPROXIMATION.
# not perfection.


## Small learning rate -> slow but safe learning
## Large learning rate -> faster but may overshoot and diverge
