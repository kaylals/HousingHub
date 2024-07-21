import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# environment variables
categorical_columns = ['Type', 'Stat']
input_dir = 'data/individual_level/raw_csv_700'
output_dir = 'data/individual_level/processed/encoded_700.csv'
files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.csv')]
dfs = [pd.read_csv(file) for file in files]
combined_df = pd.concat(dfs, ignore_index=True)

# logger.info(f"Columns in the DataFrame: {combined_df.columns.tolist()}")

# Step 2: drop constant columns
constant_columns = [col for col in combined_df.columns if combined_df[col].nunique() == 1]
if 'Address' in combined_df.columns:
    constant_columns.append('Address')

combined_df = combined_df.drop(columns=constant_columns)


# Step 3: Convert Categorical Variables
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, drop='first'), categorical_columns)
    ],
    remainder='passthrough'  # Leave other columns unchanged
)

# Apply the transformations to the data
processed_data = preprocessor.fit_transform(combined_df)

# Get the feature names after encoding
feature_names = (preprocessor
                 .transformers_[0][1]
                 .get_feature_names_out(categorical_columns))

# Create a new DataFrame with the encoded categorical variables
encoded_df = pd.DataFrame(processed_data[:, :len(feature_names)], columns=feature_names)

# Concatenate the encoded columns with any remaining columns
non_categorical_columns = [col for col in combined_df.columns if col not in categorical_columns]
remaining_df = combined_df[non_categorical_columns].reset_index(drop=True)
final_df = pd.concat([remaining_df, encoded_df], axis=1)

# Output the final DataFrame
print(final_df.head())
logger.info(f"Final DataFrame columns: {final_df.columns.tolist()}")


final_df.to_csv(output_dir, index=False)

final_df.head().to_csv('data/individual_level/test/encoded_700_test.csv', index=False)
print(f"Encoded data saved to {output_dir}")