import numpy as np

print("===GRADIENT DESCENT: SCALAR CASE===")

# FUNCTION
def f(w):
      return w**2

# DERIVATIVE OF F(w)
def grad(w):
      return 2*w

# START WITH RANDOM W
w=np.random.randn()
print("Initial w:",w)

learning_rate=0.1

print("\n===SMALL STEPS (GOOD LEARNING) ===")
for i in range (20):
        g=grad(w)
        w=w-learning_rate*g
        print(f"Iter {i+1}: w=w{w:.4f},f(w)={f(w):.6f}")
print("CONVERGED NEAR 0")

#=============================
#LARGE LEARNING RATE EXPERIMENT
#=============================

print("\n === LARGE LEARNING RATE (overshooting problem) ===")

w=np.random.randn()
print("INITIAL w:",w)

learning_rate=1.2  #intentionally big

for i in range (10):
      g=grad(w)
      w=w-learning_rate*g
      print(f"Iter {i+1}: w={w:.4f},f(w)={f(w):.6f}")

print("NOTICE: VALUES JUMP WILDLY INSTEAD OF CONVERGING")
# Loss decreases when we move opposite to gradient because gradient points toward increasing slope.
# We want to go downhill → so we move in negative gradient direction.
