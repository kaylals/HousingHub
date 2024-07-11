import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  


# Define the folder path where the CSV files are located
folder_path = './data'

# Dictionary to store the dataframes
dataframes = {}
dataframes_normalized = {}

# Read and process the CSV files
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv') and '700 Queen' in file_name:
        file_path = os.path.join(folder_path, file_name)
        
        # Read the CSV file
        data = pd.read_csv(file_path, skiprows=10)
        
        # Read the attribute name from the first line of the file
        with open(file_path, 'r') as f:
            lines = f.readlines()
            attribute_name = lines[0].split(',')[1].strip()
            attribute_name = attribute_name.replace("700_", "")  # Remove the "700" prefix
        
        # Remove the last row of the dataframe
        data = data.iloc[:-1]
        
        # Rename the relevant column
        data.rename(columns={'700 - Queen Anne/Magnolia': attribute_name}, inplace=True)
        
        # Filter the relevant columns
        filtered_data = data.loc[:, ['Date', attribute_name]]
        
        # Convert the 'Date' column to datetime format
        filtered_data['Date'] = pd.to_datetime(filtered_data['Date'], format='%B %Y')

        # Normalize the attribute values
        scaler = MinMaxScaler()
        normalized_values = scaler.fit_transform(filtered_data[[attribute_name]])
        filtered_data_normalized = filtered_data.copy()
        filtered_data_normalized[attribute_name] = normalized_values

        # Store the processed dataframe in the dictionary
        dataframes[attribute_name] = filtered_data
        dataframes_normalized[f'{attribute_name}_normalized'] = filtered_data_normalized


combined_df = None
for key, df in dataframes.items():
    if combined_df is None:
        combined_df = df
    else:
        combined_df = pd.merge(combined_df, df, on='Date', how='outer')
# Print the consolidated dataframe
print(combined_df.head())


# Combine all dataframes into one based on the 'Date' column
combined_df_normalized = None
for key, df in dataframes_normalized.items():
    if combined_df_normalized is None:
        combined_df_normalized = df
    else:
        combined_df_normalized = pd.merge(combined_df_normalized, df, on='Date', how='outer')

# Print the consolidated dataframe
print(combined_df_normalized.head())

# Output combined_df to CSV file
output_file_path = './combined_df.csv'
combined_df.to_csv(output_file_path, index=False)

output_file_path = './combined_df_normalized.csv'
combined_df_normalized.to_csv(output_file_path, index=False)