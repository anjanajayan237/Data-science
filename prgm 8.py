from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
print("Dataset shape:", X.shape)
print("Target shape:", y.shape)
X_bmi = X[:, 2].reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X_bmi, y, test_size=0.2, random_state=42
)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_mae = mean_absolute_error(y_test, y_pred_linear)
linear_r2 = r2_score(y_test, y_pred_linear)
print("\n--- Linear Regression using BMI ---")
print("Mean Squared Error (MSE):", linear_mse)
print("Mean Absolute Error (MAE):", linear_mae)
print("R2 Score:", linear_r2)
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X, y, test_size=0.2, random_state=42
)
multiple_model = LinearRegression()
multiple_model.fit(X_train_multi, y_train_multi)
y_pred_multi = multiple_model.predict(X_test_multi)

multi_mse = mean_squared_error(y_test_multi, y_pred_multi)
multi_mae = mean_absolute_error(y_test_multi, y_pred_multi)
multi_r2 = r2_score(y_test_multi, y_pred_multi)

print("\n--- Multiple Linear Regression ---")
print("Mean Squared Error (MSE):", multi_mse)
print("Mean Absolute Error (MAE):", multi_mae)
print("R2 Score:", multi_r2)

print("\n--- Model Comparison ---")

print("Linear Regression R2 Score:",
      round(linear_r2, 4))

print("Multiple Linear Regression R2 Score:",
      round(multi_r2, 4))

plt.scatter(X_test, y_test, label="Actual Data")

sorted_indices = X_test[:, 0].argsort()

plt.plot(
    X_test[sorted_indices],
    y_pred_linear[sorted_indices],
    color="red",
    label="Regression Line"
)
plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.title("Linear Regression using BMI")
plt.legend()
plt.show()
