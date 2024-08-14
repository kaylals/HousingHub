import pandas as pd
import numpy as np
import sys
df = pd.read_csv('data/mixed_level/700_feature_engineer.csv')

df['Price'] = np.exp(df['Log Price'])

month = 1
year = 2023
filtered_df = df[df['Year'] == year]
print(len(filtered_df))
# Apply multiple conditions and select the price column
filtered_df = filtered_df[(filtered_df['Bths'] == 2) & (filtered_df['Bds'] == 3)]

# Show only the 'price' column
price_column = filtered_df['Price'].astype(int)
average_price = price_column.mean()
print(price_column.tolist())
print(average_price)
