import pandas as pd

# Make sure 'Training_Data.csv' is in the same directory as this script,
# or provide the full path to the file.
try:
    df = pd.read_csv('Training_Data.csv')
    num_reviews = len(df)
    print(f"The total number of reviews in Training_Data.csv is: {num_reviews}")
except FileNotFoundError:
    print("Error: Training_Data.csv not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")