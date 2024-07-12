import os
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# Read the combined normalized CSV file
file_path = './combined_df_normalized.csv'
data = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Store the processed dataframe in the dictionary
dataframes = {}
for column in data.columns:
    if column != 'Date':
        dataframes[column] = data[['Date', column]].rename(columns={column: 'Value'})

keys = sorted(dataframes.keys())
split1 = len(keys) // 3
split2 = 2 * len(keys) // 3

# Create a Dash application
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Interactive Line Chart Dashboard", style={'textAlign': 'center'}),
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=data['Date'].min(),
        end_date=data['Date'].max(),
        display_format='YYYY-MM-DD',
        style={'margin': '20px', 'font-size': '14px', 'width': '50%'}
    ),
    dcc.Graph(id='line-chart'),
    html.Div([
        html.Div([
            dcc.Checklist(
                id='variable-checklist-1',
                options=[{'label': k, 'value': k} for k in keys[:split1]],
                value=keys[:split1],
                style={'margin': '10px'}
            ),
            dcc.Checklist(
                id='variable-checklist-2',
                options=[{'label': k, 'value': k} for k in keys[split1:split2]],
                value=keys[split1:split2],
                style={'margin': '10px'}
            ),
            dcc.Checklist(
                id='variable-checklist-3',
                options=[{'label': k, 'value': k} for k in keys[split2:]],
                value=keys[split2:],
                style={'margin': '10px'}
            ),
        ], style={'display': 'flex', 'justifyContent': 'center'}),
    ], style={'margin': '20px'}),
])

# Define the callback to update the line chart
@app.callback(
    Output('line-chart', 'figure'),
    [
        Input('variable-checklist-1', 'value'),
        Input('variable-checklist-2', 'value'),
        Input('variable-checklist-3', 'value'),
        Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date')
    ]
)
def update_line_chart(selected_variables_1, selected_variables_2, selected_variables_3, start_date, end_date):
    selected_variables = selected_variables_1 + selected_variables_2 + selected_variables_3
    fig = go.Figure()
    
    for variable in selected_variables:
        df = dataframes[variable]
        filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Value'], mode='lines', name=variable))
    
    fig.update_layout(
        title="Trends Over Time for Queen Anne/Magnolia Area",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white",
        margin={'l': 40, 'r': 40, 't': 40, 'b': 40},
        hovermode='closest'
    )
    
    return fig

# Run the Dash application
if __name__ == '__main__':
    app.run_server(debug=True)