import pandas as pd
import plotly.graph_objects as go

# Define the path to the normalized CSV file
csv_path = 'data/aggregated_700.csv'

# Read the data from the CSV file
data = pd.read_csv(csv_path)

# Create a figure
fig = go.Figure()

# Add traces for each of the market activity indicators
fig.add_trace(go.Scatter(x=data['Date'], y=data['Pending Sales'], mode='lines', name='Pending Sales'))
fig.add_trace(go.Scatter(x=data['Date'], y=data['Closed Sales'], mode='lines', name='Closed Sales'))
fig.add_trace(go.Scatter(x=data['Date'], y=data['Homes for Sale'], mode='lines', name='Homes for Sale'))

# Update layout for better readability and styling
fig.update_layout(
    title='Market Supply and Demand Over Time',
    xaxis_title='Date',
    yaxis_title='Count',
    title_x=0.5,
    margin={'l': 40, 'r': 40, 't': 60, 'b': 40},
    height=600,
    legend_title_text='Market Activity',
)

# Save the plot as an HTML file
fig.write_html("plot/market_supply_demand.html")
print("Market Supply and Demand HTML file saved.")
