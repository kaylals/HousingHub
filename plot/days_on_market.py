import pandas as pd
import plotly.express as px

# Define the path to the normalized CSV file
normalized_csv_path = 'data/aggregated_700.csv'

# Read the normalized data from the CSV file
data = pd.read_csv(normalized_csv_path)

# Create a scatter plot of Average Sales Price vs Average Days on Market
fig = px.scatter(data, x='Average Days on Market', y='Average Sales Price', 
                 title="Price vs. Days on Market",
                 labels={'Average Days on Market': 'Average Days on Market', 'Average Sales Price': 'Average Sales Price'},
                 trendline="ols")

# Update layout for better readability
fig.update_layout(
    xaxis_title="Average Days on Market",
    yaxis_title="Average Sales Price",
    margin={'l': 40, 'r': 40, 't': 60, 'b': 40},
    height=600,
)

# Save the plot as an HTML file
fig.write_html("plot/days_on_market.html")
print("Price vs. Days on Market (Scatter Plot) HTML file saved.")
