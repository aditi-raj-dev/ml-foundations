import numpy as np

# ---------------------------------------------
# Helper Functions
# ---------------------------------------------
def generate_data():
    np.random.seed(0)
    X = np.linspace(0, 10, 50)
    y = 3 * X + 4 + np.random.randn(*X.shape)  # y = 3x + 4 + noise
    return X, y

def compute_loss(X, y, m, c):
    y_pred = m * X + c
    return np.mean((y - y_pred) ** 2)

def gradient_descent(X, y, lr, m_init=0, c_init=0, iterations=50, verbose=False):
    m = m_init
    c = c_init
    losses = []

    for i in range(iterations):
        y_pred = m * X + c
        dm = (-2/len(X)) * np.sum(X * (y - y_pred))
        dc = (-2/len(X)) * np.sum(y - y_pred)

        m = m - lr * dm
        c = c - lr * dc

        loss = compute_loss(X, y, m, c)
        losses.append(loss)

        if verbose and i % 10 == 0:
            print(f"Iter {i}: m={m:.4f}, c={c:.4f}, loss={loss:.6f}")

    return m, c, losses


# ==========================================================
# 1️⃣ EXPERIMENT — DIFFERENT LEARNING RATES
# ==========================================================
print("\n================= LEARNING RATE EXPERIMENT =================")

X, y = generate_data()

learning_rates = [0.0001, 0.01, 1.0]

for lr in learning_rates:
    print(f"\nLearning Rate = {lr}")
    m, c, losses = gradient_descent(X, y, lr, iterations=80)

    print("Initial Loss:", round(losses[0], 6))
    print("Middle Loss:", round(losses[len(losses)//2], 6))
    print("Final Loss:", round(losses[-1], 6))

    if lr == 0.0001:
        print("→ Behaviour: Very slow learning, loss decreases tiny bit each step")
    elif lr == 0.01:
        print("→ Behaviour: Stable + smooth convergence, best learning rate")
    else:
        print("→ Behaviour: Unstable / Diverging or bouncing around")


# ==========================================================
# 2️⃣ EXPERIMENT — DIFFERENT INITIAL VALUES
# ==========================================================
print("\n================= INITIAL VALUE EXPERIMENT =================")

print("\nStarting from BIG POSITIVE values (m=50, c=50)")
m, c, losses_big_pos = gradient_descent(X, y, lr=0.01, m_init=50, c_init=50, iterations=60)
print("Initial Loss:", losses_big_pos[0])
print("Final Loss:", losses_big_pos[-1])
print("→ Result: Still converges, but starts extremely high and takes time")


print("\nStarting from BIG NEGATIVE values (m=-50, c=-50)")
m, c, losses_big_neg = gradient_descent(X, y, lr=0.01, m_init=-50, c_init=-50, iterations=60)
print("Initial Loss:", losses_big_neg[0])
print("Final Loss:", losses_big_neg[-1])
print("→ Result: Also converges, gradient direction correct, just longer")


# ==========================================================
# 3️⃣ FAILURE CASE — INSANE LEARNING RATE
# ==========================================================
print("\n================= FAILURE CASE =================")

print("\nTrying insane learning rate = 10")
m, c, losses_fail = gradient_descent(X, y, lr=10, iterations=20)

print("First Loss:", losses_fail[0])
print("Last Loss:", losses_fail[-1])
print("→ Result: Loss explodes instead of decreasing = Divergence")
