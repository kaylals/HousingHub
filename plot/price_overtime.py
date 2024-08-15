import pandas as pd
import plotly.express as px

# Define the path to the normalized CSV file
normalized_csv_path = 'data/aggregated_700.csv'

# Read the normalized data from the CSV file
data = pd.read_csv(normalized_csv_path)

# Create the line chart for Average Sales Price over time
fig = px.line(data, x='Date', y='Average Sales Price', 
              title='Trend Over Time: Average Sales Price')

# Update layout to match your dashboard style
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Average Sales Price",
    title_x=0.5,
    margin={'l': 40, 'r': 40, 't': 60, 'b': 40},
    height=600,
)

# Save the plot as an HTML file
fig.write_html("plot/Avg_sales_price.html")
print("Average Sales Price trend HTML file saved.")
