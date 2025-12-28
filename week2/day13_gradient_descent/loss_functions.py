import numpy as np

#==================================
#LOSS FUNCTIONS IN ML
#==================================

#GROUND TRUTH VALUES
y_true=np.array([2,4,6,8])

#PREDICTED VALUES (good model)
y_pred_good = np.array([2.5,3.8,5.5,8.2])

#PREDICTED VALUES (bad model)
y_pred_bad = np.array([1,7,2,10])


#------------------------------------
#1. MEAN SQUARED ERROR
#------------------------------------

def mse(y_true,y_pred):
      errors = y_true-y_pred
      squared = errors **2
      mean_sq_error =  np.mean(squared)

      print("\n MSE CALCULATION:")
      print("ERRORS:",errors)
      print("SQUARED ERRORS:", squared)
      print("MSE:",mean_sq_error)

      return mean_sq_error


#-----------------------------------
#2. MEAN ABSOLUTE ERROR
#-----------------------------------

def mae(y_true,y_pred):
      abs_errors = np.abs(y_true -y_pred)
      mean_abs_error = np.mean(abs_errors)

      print("\n MAE CALCULATION:")
      print("ABSOLUTE ERRORS:", abs_errors)
      print("MAE:",mean_abs_error)

      return mean_abs_error


#-----------------------------------
#3. DERIVATIVE OF MSE (IMPORTANT)
#-----------------------------------

def mse_derivative(y_true,y_pred):
      #FORMULA: d/dw = 2*(y_pred -y_true)/n
      grad = 2*(y_pred -y_true)/len(y_true)

      print("\n Derivative of MSE (GRADIENT):")
      print(grad)
      print("MEANING: tells how fast and in which direction loss changes")

      return grad


#==================================
#RUN
#==================================
print("\n===== GOOD MODEL =====")
mse(y_true,y_pred_good)
mae(y_true,y_pred_good)
mse_derivative(y_true,y_pred_good)

print("\n===== BAD MODEL =====")
mse(y_true,y_pred_bad)
mae(y_true,y_pred_bad)
mse_derivative(y_true,y_pred_bad)

print("""
INTUITION:
MSE punishes big mistakes more because squaring magnifies large errors.
MAE treats all mistakes fairly because it uses absolute value.
""")



