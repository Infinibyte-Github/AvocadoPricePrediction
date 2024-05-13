"""
Preprocessing of dataset
Author: Bart Schelpe
Filename: 1_Preprocessing.py
Dataset: Avocado Prices
Link: https://www.kaggle.com/datasets/neuromusic/avocado-prices
Code based on: Python4AI PowerPoint presentations and documentation of varying Python packages
"""

# import packages
import pandas as pd

# read data from csv file
df = pd.read_csv("../data/avocado.csv")

# drop the columns that are not needed for the model
not_needed = ["Unnamed: 0", "Date", "region"]
df = df.drop(columns=not_needed, axis=1)

# drop incomplete rows
df = df.dropna()

# save the dataframe to a pickle file
df.to_pickle("../data/df.pickle")
