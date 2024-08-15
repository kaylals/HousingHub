import pandas as pd
import plotly.express as px

# Define the path to the normalized CSV file
csv_path = 'data/aggregated_700.csv'

# Read the data from the CSV file
data = pd.read_csv(csv_path)

# Define price ranges
bins = [0, 200000, 400000, 600000, 800000, 1000000, float('inf')]
labels = ['<200K', '200K-400K', '400K-600K', '600K-800K', '800K-1M', '>1M']

# Categorize the data into price ranges
data['Price Range'] = pd.cut(data['Median Sales Price'], bins=bins, labels=labels)

# Group the data by price range
price_range_distribution = data['Price Range'].value_counts().sort_index()

# Create a Pie/Donut chart
fig = px.pie(values=price_range_distribution.values, names=price_range_distribution.index, hole=0.4, 
             title="Sales Distribution by Price Range")

# Update layout for better readability
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(title_x=0.5)

# Save the plot as an HTML file
fig.write_html("plot/sales_distribution_by_price_range.html")
print("Sales Distribution by Price Range HTML file saved.")
