"""
Training a K-Nearest Neighbors model
Author: Bart Schelpe
Filename: 4_3_TrainingKNN.py
Dataset: Avocado Prices
Link: https://www.kaggle.com/datasets/neuromusic/avocado-prices
Code based on: Python4AI PowerPoint presentations and documentation of varying Python packages
"""

# import packages
import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np
from sklearn.neighbors import KNeighborsRegressor
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

# # Test different values for k
# error_rate = []
# for i in range(1, 40):
#     knn = KNeighborsRegressor(n_neighbors=i)
#     knn.fit(X_train, y_train)
#     pred_i = knn.predict(X_test)
#     error_rate.append(np.sqrt(mean_squared_error(y_test, pred_i)))
#     print(f"K = {i} -> Error = {error_rate[-1]}")
#
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
#          markerfacecolor='red', markersize=10)
# plt.title('Error Rate vs. K Value')
# plt.xlabel('K')
# plt.ylabel('Error Rate')
# plt.show()

# Initialize the KNN classifier k=3 seems to be the best value,
# default weights (uniform) also seemed to be the best, algorithm and p-value didn't seem to matter
knn_classifier = KNeighborsRegressor(n_neighbors=3)

# Train the KNN classifier on the training data
knn_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn_classifier.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# save the model
joblib.dump(knn_classifier, "../models/KNNModel.joblib")

# Plot the results
Maxplot = max(max(y_test), max(y_pred))
Minplot = min(min(y_test), min(y_pred))
plt.plot([Minplot, Maxplot], [Minplot, Maxplot], color='red')
plt.scatter(y_test, y_pred, color="red")
plt.axline((0, 0), slope=1)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted (K-Nearest Neighbors)')
plt.annotate(f"K = 3\nMSE = {mse:.4f}\nR2 = {r2:.2f}", xy=(0.1, 0.85), xycoords='axes fraction')
plt.show()
