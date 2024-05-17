"""
Training a Neural Network model
Author: Bart Schelpe
Filename: 4_4_TrainingNeuralNetwork.py
Dataset: Avocado Prices
Link: https://www.kaggle.com/datasets/neuromusic/avocado-prices
Code based on: Python4AI PowerPoint presentations and documentation of varying Python packages
"""

# import packages
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
import time

# time the training
start = time.time()

# read dataframes from pickle files
dfTrain = pd.read_pickle("../data/dfTrainMinMaxScaler.pickle")
dfTest = pd.read_pickle("../data/dfTestMinMaxScaler.pickle")

# Split the dataframes into X and y
X_train = dfTrain.drop(columns=['AveragePrice'])
y_train = dfTrain['AveragePrice']

X_test = dfTest.drop(columns=['AveragePrice'])
y_test = dfTest['AveragePrice']

# Get the number of features
input_dim = X_train.shape[1]

# Define the model
model = Sequential([
    Dense(128, input_dim=input_dim, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1)
])

# Compile the model with appropriate loss function and optimizer
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

# Define the TensorBoard callback
log_dir = "logs/fit/"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=32, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10), tensorboard_callback])

# Evaluate the model
test_loss = model.evaluate(X_test, y_test)

print('Test Loss:', test_loss)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)


# Save the model
model.save("../models/NeuralNetworkModel.keras")

# Plot the results
# Plot the results
Maxplot = float(max(max(y_test), max(y_pred)))
Minplot = float(min(min(y_test), min(y_pred)))
plt.plot([Minplot, Maxplot], [Minplot, Maxplot], color='red')
plt.scatter(y_test, y_pred, color="red")
plt.axline((0, 0), slope=1)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted (Neural Network)')
plt.annotate(f"MSE = {mse:.4f}\nR2 = {r2:.2f}", xy=(0.1, 0.85), xycoords='axes fraction')
plt.show()

# print the time it took to train the model
print("Training time:", time.time() - start)
