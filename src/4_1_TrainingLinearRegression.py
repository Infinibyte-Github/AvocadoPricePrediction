"""
Training a Linear Regression model
Author: Bart Schelpe
Filename: 4_1_TrainingLinearRegression.py
Dataset: Avocado Prices
Link: https://www.kaggle.com/datasets/neuromusic/avocado-prices
Code based on: Python4AI PowerPoint presentations and documentation of varying Python packages
"""

# import packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import joblib

# Load the scaled datasets
dfTrain = pd.read_pickle('../data/dfTrainMinMaxScaler.pickle')
dfTest = pd.read_pickle('../data/dfTestMinMaxScaler.pickle')

# Split the datasets into features and target
X_train = dfTrain.drop(columns=['AveragePrice'])
y_train = dfTrain['AveragePrice']

X_test = dfTest.drop(columns=['AveragePrice'])
y_test = dfTest['AveragePrice']

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# save the model
joblib.dump(model, "../models/LinearRegressionModel.joblib")

# Plot the results
Maxplot = max(max(y_test), max(y_pred))
Minplot = min(min(y_test), min(y_pred))
plt.plot([Minplot, Maxplot], [Minplot, Maxplot], color='red')
plt.scatter(y_test, y_pred, color="red")
plt.axline((0, 0), slope=1)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted (Linear Regression)')
plt.show()
