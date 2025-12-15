##################### Section 1 (Data Preprocessing) ####################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score

# Load dataset
df = pd.read_csv('bankloan.csv')
print(df)

# Null values
print("Null values:\n", df.isnull().sum())

# Duplicate rows
print("Duplicate rows:", df.duplicated().sum())

# Basic stats
print(df.describe())

# Fix negative Experience
df['Experience'] = df['Experience'].clip(lower=0)
print(df.describe())

# Split features and target
y = df.iloc[:, 9]
x = df.drop(df.columns[9], axis=1)
x = x.drop("ID", axis=1)

# One-hot encoding
x = pd.get_dummies(x, columns=['Education'], prefix='Edu')
x = pd.get_dummies(x, columns=['ZIP.Code'], prefix='ZIP')

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

# Scale numerical features
num_features = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Mortgage']
scaler = MinMaxScaler()
x_train[num_features] = scaler.fit_transform(x_train[num_features])
x_test[num_features] = scaler.transform(x_test[num_features])

print(x_train.head())


###################### Section 2 (Linear Regression Model) #######################

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize model
lr = LinearRegression()

# Train model
lr.fit(x_train, y_train)

# Predict
y_pred = lr.predict(x_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("----------- Linear Regression Results -----------")
print("MSE :", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

# 5-Fold Cross-Validation
cv_scores = cross_val_score(lr, x_train, y_train, cv=5, scoring='r2')
print("5-Fold CV R² scores:", cv_scores)
print("Mean CV R²:", np.mean(cv_scores))

# Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted (Linear Regression)")
plt.grid()
plt.show()


############################ Section 3 (KNN Regression Model) ############################

from sklearn.neighbors import KNeighborsRegressor

# Initialize KNN Regressor
knn = KNeighborsRegressor(n_neighbors=5)

# Train the model
knn.fit(x_train, y_train)

# Predict
y_pred_knn = knn.predict(x_test)

# Evaluation
mse_knn = mean_squared_error(y_test, y_pred_knn)
rmse_knn = np.sqrt(mse_knn)
r2_knn = r2_score(y_test, y_pred_knn)

print("----------- KNN Regressor Results -----------")
print("MSE :", mse_knn)
print("RMSE:", rmse_knn)
print("R² Score:", r2_knn)

# 5-Fold Cross-Validation
cv_scores_knn = cross_val_score(knn, x_train, y_train, cv=5, scoring='r2')
print("5-Fold CV R² scores (KNN):", cv_scores_knn)
print("Mean CV R² (KNN):", np.mean(cv_scores_knn))

# Actual vs Predicted values
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_knn, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted (KNN Regressor)")
plt.grid()
plt.show()
