import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data/mixed_level/700_feature_engineer.csv', low_memory=False)

# Handle infinite values by replacing them with NaN, then drop any rows with NaN values
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Ensure all data types are numeric, VIF cannot be calculated if any categorical data is included
df_numeric = df.select_dtypes(include=[np.number])

# Recalculate VIF
VIF_data = pd.DataFrame()
VIF_data['feature'] = df_numeric.columns
VIF_data['VIF'] = [variance_inflation_factor(df_numeric.values, i) for i in range(df_numeric.shape[1])]

# Print the VIF DataFrame to see all VIF values
print(VIF_data)

# Identify features with infinite VIF values
inf_vif_features = VIF_data[VIF_data['VIF'] == float('inf')]['feature']
print("Features with infinite VIF values:")
print(inf_vif_features)

# Calculate the correlation matrix
correlation_matrix = df_numeric.corr()

# Check correlations for the features with infinite VIF values
print("Correlations of features with infinite VIF values:")
for feature in inf_vif_features:
    print(f"\nCorrelations for {feature}:")
    print(correlation_matrix[feature][inf_vif_features].sort_values(ascending=False))

# Identify pairs of features with high correlations among the infinite VIF features
high_corr_pairs = []
threshold = 0.9  # You can adjust the threshold based on your requirement
for i in range(len(inf_vif_features)):
    for j in range(i + 1, len(inf_vif_features)):
        feature1 = inf_vif_features.iloc[i]
        feature2 = inf_vif_features.iloc[j]
        corr_value = correlation_matrix.loc[feature1, feature2]
        if abs(corr_value) >= threshold:
            high_corr_pairs.append((feature1, feature2, corr_value))

print("Highly correlated pairs among infinite VIF features:")
print(high_corr_pairs)


# Remove the less relevant feature from each pair
df_reduced = df.drop(columns=['Bths', 'Bds', 'Type_COND'])  # Adjust based on your decision

# Recalculate VIF to check if the multicollinearity issue is resolved
df_numeric_reduced = df_reduced.select_dtypes(include=[np.number])

VIF_data_reduced = pd.DataFrame()
VIF_data_reduced['feature'] = df_numeric_reduced.columns
VIF_data_reduced['VIF'] = [variance_inflation_factor(df_numeric_reduced.values, i) for i in range(df_numeric_reduced.shape[1])]
print(VIF_data_reduced)
