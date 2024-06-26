"""
Inspect the features of the dataset and their relationship with the target variable.
Author: Bart Schelpe
Filename: 4_4_TrainingNeuralNetwork.py
Dataset: Avocado Prices
Link: https://www.kaggle.com/datasets/neuromusic/avocado-prices
Code based on: Python4AI PowerPoint presentations and documentation of varying Python packages
"""

# import packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# read dataframe from pickle file
df = pd.read_pickle("../data/df.pickle")

# Drop genre column
df = df.drop(columns=['type'])

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap (Unscaled)')
plt.show()

# Split the dataset into features and target variable
X = df.drop(columns=['AveragePrice'])
y = df['AveragePrice']

# Define the number of rows and columns for subplots
num_rows = 4  # You can adjust the number of rows and columns as needed
num_cols = 3

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 12))

# Flatten the axes array
axes = axes.flatten()

# Iterate over each feature
for i, feature in enumerate(X.columns):
    # Plot the feature against the target variable
    axes[i].scatter(X[feature], y, alpha=0.5)
    axes[i].set_title(f'{feature} vs Target')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Popularity')

    # Add 45-degree line
    axes[i].axline((0, 0), slope=1, color='red', linestyle='--')

fig.suptitle('Features vs Popularity (Unscaled)')
# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

# same thing but with the scaled values
# read dataframe from pickle file
df = pd.read_pickle("../data/dfTrainMinMaxScaler.pickle")

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap (Scaled)')
plt.show()

# Split the dataset into features and target variable
X = df.drop(columns=['AveragePrice'])
y = df['AveragePrice']

# Define the number of rows and columns for subplots
num_rows = 4  # You can adjust the number of rows and columns as needed
num_cols = 3

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 12))

# Flatten the axes array
axes = axes.flatten()

# Iterate over each feature
for i, feature in enumerate(X.columns):
    # Plot the feature against the target variable
    axes[i].scatter(X[feature], y, alpha=0.5)
    axes[i].set_title(f'{feature} vs Target')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Popularity')

    # Add 45-degree line
    axes[i].axline((0, 0), slope=1, color='red', linestyle='--')

fig.suptitle('Features vs AveragePrice (Scaled)')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

#histogram of the target variable
plt.figure(figsize=(10, 6))
sns.histplot(df['AveragePrice'], kde=True)
plt.xlabel('AveragePrice')
plt.ylabel('Count')
plt.title('Distribution of AveragePrice')
plt.show()
