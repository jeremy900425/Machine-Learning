import pandas as pd

# Load the training data
training_data = pd.read_csv('training_data.csv')

# Fill missing values with the median of each column
training_data_filled = training_data.fillna(training_data.median())

# Save the filled data (optional)
training_data_filled.to_csv('filled_training_data.csv', index=False)
