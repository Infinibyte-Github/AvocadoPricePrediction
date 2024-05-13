"""
Scaling of the dataset for training purposes
Author: Bart Schelpe
Filename: 3_Scaling.py
Dataset: Avocado Prices
Link: https://www.kaggle.com/datasets/neuromusic/avocado-prices
Code based on: Python4AI PowerPoint presentations and documentation of varying Python packages
"""

# import packages
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

# read dataframes from pickle files
dfTrain = pd.read_pickle("../data/dfTrain.pickle")
dfTest = pd.read_pickle("../data/dfTest.pickle")

# Initialize OneHotEncoder
encoder = OneHotEncoder()

# Fit and transform the 'type' column
encoder.fit(dfTrain[['type']])

# Transform the dataframe
encoded_data_train = encoder.transform(dfTrain[['type']])
encoded_data_test = encoder.transform(dfTest[['type']])

# One-hot encode the 'type' column
encoded_df_train = pd.DataFrame(encoded_data_train.toarray(), columns=encoder.get_feature_names_out(['type']))
encoded_df_test = pd.DataFrame(encoded_data_test.toarray(), columns=encoder.get_feature_names_out(['type']))

# Concatenate the encoded DataFrame with the original DataFrame
dfTrain = pd.concat([dfTrain.reset_index(drop=True), encoded_df_train], axis=1)
dfTest = pd.concat([dfTest.reset_index(drop=True), encoded_df_test], axis=1)

# Drop the original 'type' column
dfTrain.drop(columns=['type'], inplace=True)
dfTest.drop(columns=['type'], inplace=True)



# save the dataframes to new pickle files
dfTrain.to_pickle("../data/dfTrainMinMaxScaler.pickle")
dfTest.to_pickle("../data/dfTestMinMaxScaler.pickle")
