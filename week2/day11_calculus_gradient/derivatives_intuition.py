import numpy as np

print("===DERIVATIVE INTUITION ===")

# FUNCTION f(x) =x^2
def f(x):
    return x**2

# NUMERICAL DERIVATIVE USING FINITE DIFFERENCE
def numerical_derivative(x):
   h=1e-5  # very small change
   return(f(x+h)-f(x))/h

#TEST VALUES
x_values =[-3,-1,0,1,3]

for x in x_values:
   slope = numerical_derivative(x)
   print(f"x={x},f(x)={f(x)},slope={slope}")

   if slope > 0:
           print("Meaning: Positive slope -> function increasing here")
   elif slope < 0:
           print("Meaning: Negative slope -> function decreasing here ")
   else:
           print("Meaning: Zero slope -> flat region (possible minimum/maximum)")

   print("--------------")`

# Gradient shows steepest uphill direction. Moving opposite ensures we always move downhill toward minimum.
