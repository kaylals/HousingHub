import pandas as pd
import plotly.graph_objects as go

# Define the path to the normalized CSV file
normalized_csv_path = 'data/aggregated_700.csv'

# Read the normalized data from the CSV file
data = pd.read_csv(normalized_csv_path)

# Create the line chart for Inventory Trends
fig = go.Figure()

fig.add_trace(go.Scatter(x=data['Date'], y=data['Months Supply of Inventory (Closed)'],
                         mode='lines', name='Closed Inventory'))
fig.add_trace(go.Scatter(x=data['Date'], y=data['Months Supply of Inventory (Pending)'],
                         mode='lines', name='Pending Inventory'))

# Update layout to match your dashboard style
fig.update_layout(
    title='Inventory Trends',
    xaxis_title="Date",
    yaxis_title="Months Supply of Inventory",
    title_x=0.5,
    margin={'l': 40, 'r': 40, 't': 60, 'b': 40},
    height=600,
)

# Save the plot as an HTML file
fig.write_html("plot/inventory_trends.html")
print("Inventory Trends HTML file saved.")
