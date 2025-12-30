

from dataset import X, y
from train_test_split import train_test_split
from linear_regression_scratch import LinearRegressionScratch
from metrics import mse, rmse, mae, r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegressionScratch(learning_rate=0.001)
model.fit(X_train, y_train, epochs=1500)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("\n----- TRAIN PERFORMANCE -----")
print("MSE:", mse(y_train, y_train_pred))
print("RMSE:", rmse(y_train, y_train_pred))
print("MAE:", mae(y_train, y_train_pred))
print("R2:", r2_score(y_train, y_train_pred))

print("\n----- TEST PERFORMANCE -----")
print("MSE:", mse(y_test, y_test_pred))
print("RMSE:", rmse(y_test, y_test_pred))
print("MAE:", mae(y_test, y_test_pred))
print("R2:", r2_score(y_test, y_test_pred))
