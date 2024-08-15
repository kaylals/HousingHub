import pandas as pd
import plotly.express as px

# Define the path to the normalized CSV file
normalized_csv_path = 'data/aggregated_700.csv'

# Read the normalized data from the CSV file
data = pd.read_csv(normalized_csv_path)

# Create a histogram for Price Per Square Foot
fig = px.histogram(data, x='Median Price Per Square Foot', nbins=30,
                   title='Price Per Square Foot Distribution',
                   labels={'Median Price Per Square Foot': 'Price Per Square Foot', 
                           'count': 'Number of Properties'})

# Update layout for better readability
fig.update_layout(
    xaxis_title="Price Per Square Foot",
    yaxis_title="Number of Properties",
    margin={'l': 40, 'r': 40, 't': 60, 'b': 40},
    height=600,
)

# Save the plot as an HTML file
fig.write_html("plot/price_per_square_foot_distribution.html")
print("Price Per Square Foot Distribution histogram HTML file saved.")
