"""
Training a Neural Network model using keras tuner
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
from tensorflow import keras
import keras_tuner as kt

# read dataframes from pickle files
dfTrain = pd.read_pickle("../data/dfTrainMinMaxScaler.pickle")
dfTest = pd.read_pickle("../data/dfTestMinMaxScaler.pickle")

# Split the dataframes into X and y
X_train = dfTrain.drop(columns=['AveragePrice'])
y_train = dfTrain['AveragePrice']

X_test = dfTest.drop(columns=['AveragePrice'])
y_test = dfTest['AveragePrice']

input_dim = X_train.shape[1]


def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))  # Define input shape using Input layer
    model.add(keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu',
                                 input_shape=(input_dim,)))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='mse',
                  metrics=['mse'])
    return model


tuner = kt.RandomSearch(build_model,
                        objective='val_loss',
                        max_trials=5,
                        directory='randomsearch',
                        project_name='AvocadoPricePrediction')

# Let the tuner search for the best hyperparameters
tuner.search(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32,
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

model = tuner.hypermodel.build(best_hps)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=32,
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])

# Evaluate the model
test_loss = model.evaluate(X_test, y_test)

print('Test Loss:', test_loss)

# Make predictions
y_pred = model.predict(X_test)

# Save the model
model.save("../models/NeuralNetworkModel.keras")

# Plot the results
# Plot the results
Maxplot = float(max(max(y_test), max(y_pred)))
Minplot = min(min(y_test), min(y_pred))
plt.plot([Minplot, Maxplot], [Minplot, Maxplot], color='red')
plt.scatter(y_test, y_pred, color="red")
plt.axline((0, 0), slope=1)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted (Neural Network)')
plt.show()
