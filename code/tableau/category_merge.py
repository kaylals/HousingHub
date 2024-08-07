import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv('data/tableau_stat/700_fe.csv')

# Step 2: Filter out rows where all three category columns are 0
df = df[(df['Size Category_Large'] != 0) | (df['Size Category_Small'] != 0) | (df['Size Category_Medium'] != 0)]

# Step 3: Create the two new columns
def map_category(row):
    if row['Size Category_Large'] == 1:
        return 1, 'large'
    elif row['Size Category_Medium'] == 1:
        return 2, 'medium'
    elif row['Size Category_Small'] == 1:
        return 3, 'small'
    else:
        return 4, 'none'

df[['Category Number', 'Category Label']] = df.apply(map_category, axis=1, result_type='expand')

# Step 4: Save the DataFrame back to a CSV file (optional)
df.to_csv('data/tableau_stat/700_fe.csv', index=False)