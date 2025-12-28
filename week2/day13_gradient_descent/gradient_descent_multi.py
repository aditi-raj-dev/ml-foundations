import numpy as np

#=================================
#DATASET
#y=2x+1(But we won't tell GD this)
#=================================

X=np.array([1,2,3,4], dtype =float)
y=np.array([3,5,7,9],dtype=float)

# Convert X into 2D (matrix form)
X=X.reshape(-1,1)

#=============================
#PARAMETERS
#=============================
m=np.random.randn() #slope
c=np.random.randn() #intercept

lr=0.01
epochs=100

def predict(X):
      return m*X+c

def mse(y_true, y_pred):
      return np.mean((y_true-y_pred)**2)

print("INITIAL m:", m)
print("INITIAL c:",c)

for i in range (epochs):
    y_pred=predict(X)
    # ===========================
    # GRADIENTS (VERY IMPORTANT)
    # ===========================
    dm = (-2 / len(X)) * np.sum(X * (y - y_pred))
    dc = (-2 / len(X)) * np.sum(y - y_pred)

    # ===========================
    # UPDATE RULE
    # Move opposite of gradient
    # ===========================
    m = m - lr * dm
    c = c - lr * dc

    loss = mse(y, y_pred)

    if i % 10 == 0:
        print(f"Epoch {i}/{epochs}: Loss={loss:.6f}, m={m:.4f}, c={c:.4f}")

print("\nFINAL RESULTS")
print("m =", m)
print("c =", c)
print("Final Loss =", mse(y, predict(X)))

"""
INTUITION:

We are learning the LINE that best fits data.

m controls SLOPE (tilt of line)
c controls SHIFT (how high/low the line is)

Gradient tells:
→ How wrong we are
→ Which direction to move m and c
→ How big the correction should be

Each epoch:
We slightly adjust m and c
Loss reduces
Line becomes better
Model "learns"
"""

# Gradient tells us which direction to move to reduce loss
# We update parameters opposite to gradient because we want to minimize loss

# Intuition: Loss decreases because parameters move opposite to gradient direction

