import pandas as pd


df = pd.read_csv('data/mixed_level/feature_engineer_700.csv')

# Count all recordings (rows) in the DataFrame
total_recordings = len(df)

# Alternative method using shape
# total_recordings = df.shape[0]

print(f"Total number of recordings: {total_recordings}")

# Check for None (NaN) values in each column
null_count_per_column = df.isnull().sum()

# Filter to only columns with NaN values and display them
columns_with_nulls = null_count_per_column[null_count_per_column > 0]
print(columns_with_nulls)
