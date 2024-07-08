import os
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# Define the folder path where the CSV files are located
folder_path = './data'

# Dictionary to store the dataframes
dataframes = {}

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
        
        # Store the processed dataframe in the dictionary
        dataframes[attribute_name] = filtered_data
    
keys = sorted(dataframes.keys())
split1 = len(keys) // 3
split2 = 2 * len(keys) // 3



# Create a Dash application
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Interactive Line Chart Dashboard", style={'textAlign': 'center'}),
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=min([df['Date'].min() for df in dataframes.values()]),
        end_date=max([df['Date'].max() for df in dataframes.values()]),
        display_format='YYYY-MM-DD',
        style={'margin': '20px', 'font-size': '14px', 'width': '50%'}
    ),
    dcc.Graph(id='line-chart'),
    html.Div([
        # html.Label("Select Variables:", style={'fontSize': '17px', 'textAlign': 'center'}),
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
        fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df[variable], mode='lines', name=variable))
    
    # Update the y-axis title and tick labels based on the number of selected variables
    show_yticks = True if len(selected_variables) == 1 else False
    yaxis = dict(tickvals=None if not show_yticks else None)
    
    fig.update_layout(
        title="Trends Over Time for Queen Anne/Magnolia Area",
        xaxis_title="Date",
        yaxis_title="Value",
        yaxis=yaxis,
        template="plotly_white",
        margin={'l': 40, 'r': 40, 't': 40, 'b': 40},
        hovermode='closest'
    )
    
    return fig

# Run the Dash application
if __name__ == '__main__':
    app.run_server(debug=True)