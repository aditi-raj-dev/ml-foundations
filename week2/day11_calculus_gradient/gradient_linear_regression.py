import numpy as np

# FUNCTION
def f(x):
    return x**2+4*x+6  # simple parabola

# DERIVATIVE
def df(x):
    return 2*x+4

x=6 # starting point
lr=0.1 # learning rate

for i in range (10):
     grad=df(x)
     x=x-lr*grad
     print(f"Iter {i+1}: x={x:.4f},f(x)={f(x):.6f}")