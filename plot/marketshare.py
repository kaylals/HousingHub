import pandas as pd
import plotly.express as px

# Define the path to the normalized CSV file
normalized_csv_path = 'data/aggregated_700.csv'

# Read the normalized data from the CSV file
data = pd.read_csv(normalized_csv_path)

# Select a specific time period (e.g., the most recent month) for the analysis
# Assuming the 'Date' column is in datetime format and sorted, we'll select the latest date
latest_data = data[data['Date'] == data['Date'].max()]

# Extract relevant data
market_activity = {
    'New Listings': latest_data['New Listings'].sum(),
    'Closed Sales': latest_data['Closed Sales'].sum(),
    'Pending Sales': latest_data['Pending Sales'].sum()
}

# Create a Pie/Donut chart
fig = px.pie(values=market_activity.values(), names=market_activity.keys(), hole=0.4, 
             title="Market Share: New Listings vs Closed Sales vs Pending Sales")

# Update layout to make it look more like a donut chart
fig.update_traces(textposition='inside', textinfo='percent+label')

# Save the plot as an HTML file
fig.write_html("plot/market_share.html")
print("Market Share (Donut Chart) HTML file saved.")
