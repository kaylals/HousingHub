import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from sklearn.linear_model import LinearRegression
import numpy as np

# Define the path to the normalized CSV file
normalized_csv_path = 'data/aggregated_700_normalized.csv'

# Read the normalized data from the CSV file
data = pd.read_csv(normalized_csv_path)

# Drop the 'Date' column for correlation computation
data_without_date = data.drop(columns=['Date'])

# Compute the correlation matrix
correlation_matrix = data_without_date.corr().round(2)

# Split the variable list into three parts for better layout
variables = correlation_matrix.columns.tolist()
split1 = len(variables) // 3
split2 = 2 * len(variables) // 3

# Create a Dash application
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Correlation Matrix", style={'textAlign': 'center'}),
    
    # Correlation Matrix Heatmap
    dcc.Graph(id='heatmap'),
    html.Label("Select Variables for Correlation Matrix:", style={'fontSize': '17px', 'textAlign': 'center'}),
    html.Div([
        html.Div([
            dcc.Checklist(
                id='variable-checklist-1',
                options=[{'label': var, 'value': var} for var in variables[:split1]],
                value=variables[:split1],
                style={'margin': '10px'}
            ),
        ], style={'width': '30%', 'display': 'inline-block'}),
        html.Div([
            dcc.Checklist(
                id='variable-checklist-2',
                options=[{'label': var, 'value': var} for var in variables[split1:split2]],
                value=variables[split1:split2],
                style={'margin': '10px'}
            ),
        ], style={'width': '30%', 'display': 'inline-block'}),
        html.Div([
            dcc.Checklist(
                id='variable-checklist-3',
                options=[{'label': var, 'value': var} for var in variables[split2:]],
                value=variables[split2:],
                style={'margin': '10px'}
            ),
        ], style={'width': '30%', 'display': 'inline-block'}),
    ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'marginBottom': '20px'}),
    
    # Linear Regression Section
    html.H1("Linear Regression", style={'textAlign': 'center', 'marginTop': '40px'}),
    html.Div([
        html.Div([
            html.Label("Select X variable:", style={'fontSize': '17px', 'textAlign': 'center'}),
            dcc.Dropdown(
                id='x-variable',
                options=[{'label': col, 'value': col} for col in correlation_matrix.columns],
                value=correlation_matrix.columns[0],
                style={'margin': '10px'}
            ),
        ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            html.Label("Select Y variable:", style={'fontSize': '17px', 'textAlign': 'center'}),
            dcc.Dropdown(
                id='y-variable',
                options=[{'label': col, 'value': col} for col in correlation_matrix.columns],
                value=correlation_matrix.columns[1],
                style={'margin': '10px'}
            ),
        ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'display': 'flex', 'justifyContent': 'space-evenly'}),
    dcc.Graph(id='regression-plot', style={'marginTop': '20px'}),
])

# Define the callback to update the heatmap
@app.callback(
    Output('heatmap', 'figure'),
    [
        Input('variable-checklist-1', 'value'),
        Input('variable-checklist-2', 'value'),
        Input('variable-checklist-3', 'value')
    ]
)
def update_heatmap(selected_variables_1, selected_variables_2, selected_variables_3):
    selected_variables = selected_variables_1 + selected_variables_2 + selected_variables_3
    # Filter the correlation matrix based on the selected variables
    filtered_corr_matrix = correlation_matrix.loc[selected_variables, selected_variables]
    
    # Create the heatmap figure
    fig = px.imshow(filtered_corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
    fig.update_layout(
        title="Correlation Matrix Heatmap",
        xaxis_title="Variables",
        yaxis_title="Variables",
        margin={'l': 40, 'r': 40, 't': 40, 'b': 40}
    )
    
    return fig

# Define the callback to update the regression plot
@app.callback(
    Output('regression-plot', 'figure'),
    [Input('x-variable', 'value'), Input('y-variable', 'value')]
)
def update_regression_plot(x_variable, y_variable):
    # Prepare the data for regression
    x = data[[x_variable]].values
    y = data[y_variable].values
    
    # Perform linear regression
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    
    # Create the regression plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x.flatten(), y=y, mode='markers', name='Data Points'))
    fig.add_trace(go.Scatter(x=x.flatten(), y=y_pred, mode='lines', name='Regression Line'))
    
    fig.update_layout(
        title=f'Linear Regression: {x_variable} vs {y_variable}',
        xaxis_title=x_variable,
        yaxis_title=y_variable,
        margin={'l': 40, 'r': 40, 't': 40, 'b': 40}
    )
    
    return fig

# Run the Dash application
if __name__ == '__main__':
    app.run_server(debug=True)
