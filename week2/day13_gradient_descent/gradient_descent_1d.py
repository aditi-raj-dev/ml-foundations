import numpy as np

#=========================================
#FUNCTION: f(x)=x^2
#GOAL: FIND x that MINIMIZES this function
#Minimum occurs at x=0
#=========================================


def f(x):
    return x**2

def df(x):
    return 2*x     # derivative of x^2



#Start from a random x
x=np.random.randn()
learning_rate=0.1

print("starting x:",x)

for i in range (50):

        grad=df(x)    # slope at current point
        x=x-learning_rate*grad  #move opposite direction of slope

        if i%5==0:
             print(f"Iter {i}: x={x:.5f}, f(x)={f(x):.6f}")

print("\nFinal x:",x)
print("\nFinal f(x)",f(x))

"""
INTUITION:

f(x)=x^2 is a U-shaped curve.
Gradient(slope) tells which direction is uphill.
Positive slope-> we are on right side -> go left.
Negative slope-> we are on left side -> go right.

We always move OPPOSITE the gradient
because gradient points uphill,
but we want to go downhill(MINIMUM).
"""