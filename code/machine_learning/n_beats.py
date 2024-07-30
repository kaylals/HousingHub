# pip install torch nbeats-pytorch


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
# Load your data
input_path = "data/mixed_level/700_feature_engineer.csv"

data = pd.read_csv(input_path)
data['Stat Date'] = pd.to_datetime(data['Stat Date'])
data.set_index('Stat Date', inplace=True)

# Scale the data
column_to_delete = ['MLS','List/Sell $', 'Agg_Year', 'Agg_Month', 'Agg_Day', 'Agg_Date']
data = data.drop(columns=column_to_delete)

scaler = MinMaxScaler()
data['Log Price'] = scaler.fit_transform(data[['Log Price']])

# Split into training and validation sets
train_size = int(len(data) * 0.8)
train, val = data.iloc[:train_size], data.iloc[train_size:]


def create_dataset(data, input_steps, output_steps):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps):
        X.append(data[i:(i + input_steps), 0])
        y.append(data[(i + input_steps):(i + input_steps + output_steps), 0])
    return np.array(X), np.array(y)

input_steps = 30  # Number of past observations to use
output_steps = 7  # Number of future steps to predict

X_train, y_train = create_dataset(train.values, input_steps, output_steps)
X_val, y_val = create_dataset(val.values, input_steps, output_steps)


import torch
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


from nbeats_pytorch.model import NBeatsNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
model = NBeatsNet(
    device=device,
    stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
    forecast_length=output_steps,
    backcast_length=input_steps,
    hidden_layer_units=128
).to(device)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        _, forecast = model(x)
        loss = loss_fn(forecast, y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            _, forecast = model(x)
            val_loss += loss_fn(forecast, y).item()
    
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss/len(val_loader)}')


model.eval()
with torch.no_grad():
    x, y = torch.Tensor(X_val).to(device), torch.Tensor(y_val).to(device)
    _, predictions = model(x)
    predictions = predictions.cpu().numpy()
    
    # Inverse transform the predictions if necessary
    predictions = scaler.inverse_transform(predictions)
