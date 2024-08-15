import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Define the path to the normalized CSV file
normalized_csv_path = 'data/aggregated_700_normalized.csv'

# Read the normalized data from the CSV file
data = pd.read_csv(normalized_csv_path)

# Drop the 'Date' column for correlation computation
data_without_date = data.drop(columns=['Date'])

# Compute the correlation matrix
correlation_matrix = data_without_date.corr().round(2)

# Create the heatmap figure with similar styling
fig = px.imshow(
    correlation_matrix, 
    text_auto=True, 
    aspect="auto", 
    color_continuous_scale='RdBu_r'
)

fig.update_layout(
    title=dict(
        text="Correlation Matrix Heatmap",
        x=0.5,
        xanchor='center',
        yanchor='top',
        font=dict(size=24)
    ),
    xaxis_title="Variables",
    yaxis_title="Variables",
    xaxis=dict(tickangle=-45),  # Rotate x-axis labels to match your layout
    margin={'l': 40, 'r': 40, 't': 60, 'b': 40},
    height=600,  # Adjust height to match your desired layout
)

# # Add additional layout elements if needed, like annotations, etc.
# fig.add_annotation(
#     text="Correlation Matrix",
#     xref="paper", yref="paper",
#     x=0.5, y=1.15,
#     showarrow=False,
#     font=dict(size=28),
#     xanchor='center'
# )

# Save the plot as an HTML file with a similar layout
fig.write_html("plot/correlation_matrix_dashboard.html")

