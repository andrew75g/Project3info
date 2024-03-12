import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Assuming your data is in a CSV file named 'nfl_qb_stats.csv'
# Load the dataset
df = pd.read_csv('nfl_qb_stats.csv')

# Pre-processing
# For simplicity, let's focus on a few key features and the target variable
features = ['Pass Yds', 'Att', 'Cmp', 'TD', 'INT']
target = 'Rate'

# Splitting the data
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Experiment 1: Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
print(f"Linear Regression RMSE: {rmse_linear}")

# Experiment 2: Polynomial Regression
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
print(f"Polynomial Regression RMSE: {rmse_poly}")

# Experiment 3: Decision Tree Regression
tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
rmse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))
print(f"Decision Tree Regression RMSE: {rmse_tree}")

# Plotting the results for comparison
models = ['Linear', 'Polynomial', 'Decision Tree']
rmses = [rmse_linear, rmse_poly, rmse_tree]

plt.figure(figsize=(10, 5))
plt.bar(models, rmses, color=['blue', 'green', 'red'])
plt.title('RMSE Comparison Among Regression Models')
plt.ylabel('RMSE')
plt.show()
