import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import sys


# environment variables
input_dir = 'data/individual_level/processed/700_encoded.csv'      # test: 'data/individual_level/test/encoded_700_test.csv'
output_dir =  'data/mixed_level/700_feature_engineer.csv'    # test:  'data/individual_level/test/features_test.csv' 
input_dir_aggre = 'data/aggregated_700.csv'
# Load encoded transactional data and aggregated data
df = pd.read_csv(input_dir)
agg_df = pd.read_csv(input_dir_aggre)


df['Stat Date'] = pd.to_datetime(df['Stat Date'], errors='coerce')
agg_df['Date'] = pd.to_datetime(agg_df['Date'], errors='coerce')

agg_df_cleaned = agg_df.dropna(subset=['Date'])
agg_df['Year'] = agg_df['Date'].dt.year
agg_df['Month'] = agg_df['Date'].dt.month
agg_df['Day'] = agg_df['Date'].dt.day

# Drop observations earlier than '2006-01-01'
cutoff_date = pd.Timestamp(agg_df['Date'].min())
df = df[df['Stat Date'] >= cutoff_date]

columns_to_convert = ['List/Sell $', 'SF', 'Bds', 'Bths', 'CDOM']
for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Function to match transaction date with aggregated data
def match_date(date):
    matched = agg_df[
        (agg_df['Year'] == date.year) & 
        (agg_df['Month'] == date.month)
    ]
    if matched.empty:
        return pd.Series({col: np.nan for col in agg_columns})
    return matched.iloc[0]

# Add aggregated features
agg_columns = agg_df.columns.tolist()

for col in agg_columns:
    df[f'Agg_{col}'] = df['Stat Date'].apply(lambda x: match_date(x)[col])



# Feature Engineering steps


# 1. Create new features

# price per square foot feature
if 'List/Sell $' in df.columns and 'SF' in df.columns:
    df['Price_Per_SF'] = df['List/Sell $'] / df['SF']

# Total rooms and bathrooms to bedrooms ratio
if 'Bths' in df.columns and 'Bds' in df.columns:
    df['Total_Rooms'] = df['Bds'] + df['Bths']
    df['Bths_Bds_Ratio'] = np.where(df['Bds'] == 0, 
                                df['Bths'], 
                                df['Bths'] / df['Bds'])

# Price per bedroom
if 'List/Sell $' in df.columns and 'Bds' in df.columns:
    df['Price_per_Bedroom'] = np.where(df['Bds'] == 0, 
                                   df['List/Sell $'], 
                                   df['List/Sell $'] / df['Bds'])

# Binning the house size
if 'SF' in df.columns:
    df['Size Category'] = pd.cut(df['SF'], bins=[0, 1000, 2000, 3000, np.inf], 
                                 labels=['Small', 'Medium', 'Large', 'Extra Large'])
    
# print("Size Category value counts:")
# print(df['Size Category'].value_counts())

new_cat_vars = ['Size Category'] 

# Log transformation of price
if 'List/Sell $' in df.columns:
    df['Log Price'] = np.log(df['List/Sell $'])

# 2. Handle missing values
# Identify numeric and non-numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns
non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns

df = df.replace([np.inf, -np.inf], np.nan)
max_value = np.finfo(np.float64).max
for col in numeric_columns:
    df[col] = df[col].clip(upper=max_value)

# for col in numeric_columns:
#     if np.isinf(df[col]).any() or np.isnan(df[col]).any():
#         print(f"Column {col} contains inf or nan values")
#     if df[col].max() > np.finfo(np.float64).max:
#         print(f"Column {col} contains values too large for float64")


# print("Columns before filtering:", len(numeric_columns))
numeric_columns = [col for col in numeric_columns if not df[col].isna().all()]

# print("Columns after filtering:", len(numeric_columns))

numeric_imputer = SimpleImputer(strategy='mean')
df_numeric_imputed = pd.DataFrame(
    numeric_imputer.fit_transform(df[numeric_columns]),
    columns=numeric_columns,
    index=df.index
)

for col in set(df.columns) - set(df_numeric_imputed.columns) - set(non_numeric_columns):
    df_numeric_imputed[col] = 0  # or any other appropriate default value

non_numeric_imputer = SimpleImputer(strategy='most_frequent')
df_non_numeric_imputed = pd.DataFrame(
    non_numeric_imputer.fit_transform(df[non_numeric_columns]),
    columns=non_numeric_columns,
    index=df.index
)

# Combine the imputed dataframes
df_imputed = pd.concat([df_numeric_imputed, df_non_numeric_imputed], axis=1)

# Ensure the column order matches the original dataframe
df_imputed = df_imputed[df.columns]

# After imputation
print("NaN values after initial imputation:")
print(df_imputed.isna().sum())

# Additional imputation for specific columns
df_imputed['CDOM'] = df_imputed['CDOM'].fillna(df_imputed['CDOM'].median())

# Re-calculate relative features
df_imputed['Relative_Price_to_Avg'] = df_imputed['List/Sell $'] / df_imputed['Agg_Average Sales Price']
df_imputed['Relative_DOM_to_Avg'] = df_imputed['CDOM'] / df_imputed['Agg_Average Days on Market']
df_imputed['Relative_Price_per_SF_to_Avg'] = df_imputed['Price_Per_SF'] / df_imputed['Agg_Average Price Per Square Foot']

# Final check
print("NaN values after additional processing:")
print(df_imputed.isna().sum())


# 3. Convert date features
if 'Stat Date' in df.columns:
    df_imputed['Stat Date'] = pd.to_datetime(df_imputed['Stat Date'])
    df_imputed['Month'] = df_imputed['Stat Date'].dt.month
    df_imputed['Year'] = df_imputed['Stat Date'].dt.year
    df_imputed['DayOfWeek'] = df_imputed['Stat Date'].dt.dayofweek

# 4. Scale numerical features
numerical_features = df_imputed.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
df_imputed[numerical_features] = scaler.fit_transform(df_imputed[numerical_features])


print("Shape of df_imputed:", df_imputed.shape)
# 5. Handle outliers (example using z-score)
from scipy import stats

def remove_outliers(df, columns, z_threshold=3):
    # for col in columns:
    #     z_scores = np.abs(stats.zscore(df[col]))
    #     df = df[(z_scores < z_threshold)]
    return df

df_no_outliers = remove_outliers(df_imputed, numerical_features)

print("Shape of df_no_outliers:", df_no_outliers.shape)

# 6. encode categorical variables

# Select only the new categorical variables
new_cat_df = df_no_outliers[new_cat_vars]

# Initialize and fit OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first')

encoded_new_cats = encoder.fit_transform(new_cat_df)

feature_names = encoder.get_feature_names_out(new_cat_vars)

encoded_df = pd.DataFrame(encoded_new_cats, columns=feature_names, index=df.index)

# Concatenate the new encoded features with the original dataframe
df_final = pd.concat([df, encoded_df], axis=1)

df_final = df_final.drop(columns=new_cat_vars)

print("Shape of df_final:", df_final.shape)

# Count all recordings (rows) in the DataFrame
total_recordings = len(df_final)

# Alternative method using shape
# total_recordings = df.shape[0]

print(f"Total number of recordings: {total_recordings}")

# Check for None (NaN) values in each column
null_count_per_column = df_final.isnull().sum()

# Filter to only columns with NaN values and display them
columns_with_nulls = null_count_per_column[null_count_per_column > 0]
print(columns_with_nulls)


# # 7. Feature selection (example using correlation)
# correlation_matrix = df_final.corr()
# high_corr_features = np.where(np.abs(correlation_matrix) > 0.8)
# high_corr_features = [(correlation_matrix.index[x], correlation_matrix.columns[y]) 
#                       for x, y in zip(*high_corr_features) if x != y and x < y]

# print("\nHighly correlated features:")
# print(high_corr_features)


# # 8. mi scores

# from sklearn.feature_selection import mutual_info_regression

# # Assuming 'List/Sell $' is your target variable
# X = df_final.drop('List/Sell $', axis=1)
# y = df_final['List/Sell $']

# # Identify datetime columns
# datetime_columns = X.select_dtypes(include=[np.datetime64]).columns

# # Convert datetime columns to numerical values (e.g., timestamp)
# for col in datetime_columns:
#     X[col] = X[col].astype(int) / 10**9  # Convert to Unix timestamp


# # Identify non-numeric columns (excluding datetime which we've already handled)
# non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns

# # Encode non-numeric columns
# for col in non_numeric_columns:
#     X[col] = pd.Categorical(X[col]).codes

# # Print datatypes after encoding
# print(X.dtypes)


# # Calculate mutual information scores
# mi_scores = mutual_info_regression(X, y)

# # Create a dataframe of features and their mutual information scores
# mi_scores = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi_scores})
# mi_scores = mi_scores.sort_values('Mutual Information', ascending=False)

# print(mi_scores)


# Save the engineered features
df_final.to_csv(output_dir, index=False)
print(f"\nEngineered features saved to {output_dir}")