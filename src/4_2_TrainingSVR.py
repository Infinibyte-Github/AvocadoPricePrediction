"""
Training a Support Vector Regression model
Author: Bart Schelpe
Filename: 4_2_TrainingSVR.py
Dataset: Avocado Prices
Link: https://www.kaggle.com/datasets/neuromusic/avocado-prices
Code based on: Python4AI PowerPoint presentations and documentation of varying Python packages
"""

# import packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import joblib
import time

# time the training
start = time.time()

# Load the scaled datasets
dfTrain = pd.read_pickle('../data/dfTrainMinMaxScaler.pickle')
dfTest = pd.read_pickle('../data/dfTestMinMaxScaler.pickle')

# Split the datasets into features and target
X_train = dfTrain.drop(columns=['AveragePrice'])
y_train = dfTrain['AveragePrice']

X_test = dfTest.drop(columns=['AveragePrice'])
y_test = dfTest['AveragePrice']

# Initialize the SVR model
svm_model = SVR(kernel='poly', verbose=True)

# Train the SVR model on the training data
svm_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svm_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# save the model
joblib.dump(svm_model, "../models/SVRModel.joblib")

# Plot the results
Maxplot = max(max(y_test), max(y_pred))
Minplot = min(min(y_test), min(y_pred))
plt.plot([Minplot, Maxplot], [Minplot, Maxplot], color='red')
plt.scatter(y_test, y_pred, color="red")
plt.axline((0, 0), slope=1)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted (SVR)')
plt.annotate(f"MSE = {mse:.4f}\nR2 = {r2:.2f}", xy=(0.1, 0.85), xycoords='axes fraction')
plt.show()

# print the time it took to train the model
print("Training time:", time.time() - start)
