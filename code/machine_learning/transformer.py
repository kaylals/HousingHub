# pip install pytorch-forecasting


import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch


print("torch version: ", torch.__version__)
sys.exit()



import pytorch_forecasting
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, RMSE, SMAPE
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl





# Prepare Data
input_path = "data/mixed_level/700_feature_engineer.csv"

data = pd.read_csv(input_path)

# Scale the data
scaler = MinMaxScaler()
data['Log Price'] = scaler.fit_transform(data[['Log Price']])

data['time_idx'] = range(len(data))
data['group_id'] = 0  # Assuming single time series
# data['group_id'] = data['city_id']  # Assuming you have a city_id column



# Split into training and validation sets
train = data.iloc[:train_size]
val = data.iloc[train_size:]

# Define the training and validation datasets
max_encoder_length = input_steps
max_prediction_length = output_steps

# Create TimeSeriesDataSet
training = TimeSeriesDataSet(
    train,
    time_idx="time_idx",
    target="Log Price",
    group_ids=["group_id"],
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],
    static_reals=[],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["Log Price"],
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# Create validation set
validation = TimeSeriesDataSet.from_dataset(training, val, stop_randomization=True)

# Create DataLoaders
batch_size = 32
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

# Initialize TFT model
pl.seed_everything(42)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=1e-3,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,
    loss=SMAPE(),
    log_interval=10,
    reduce_on_plateau_patience=4
)

# Configure trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
logger = TensorBoardLogger("lightning_logs")

trainer = pl.Trainer(
    max_epochs=50,
    gpus=1 if torch.cuda.is_available() else 0,
    gradient_clip_val=0.1,
    limit_train_batches=50,
    callbacks=[lr_logger, early_stop_callback],
    logger=logger
)

# Fit the model
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# Evaluate the model
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# Make predictions
predictions = best_tft.predict(val_dataloader, return_index=True, return_decoder_lengths=True)
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])

# Calculate metrics
mae = MAE()(predictions.output, actuals)
rmse = RMSE()(predictions.output, actuals)
smape = SMAPE()(predictions.output, actuals)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"SMAPE: {smape:.4f}")

# If you need the predictions as numpy array
predictions_np = predictions.output.numpy()
actuals_np = actuals.numpy()

# Inverse transform if necessary
predictions_np = scaler.inverse_transform(predictions_np)
actuals_np = scaler.inverse_transform(actuals_np)

# Calculate final metrics on original scale
final_mae = mean_absolute_error(actuals_np, predictions_np)
final_mse = mean_squared_error(actuals_np, predictions_np)
final_rmse = np.sqrt(final_mse)

print("\nFinal Model Performance (Original Scale):")
print(f"MAE: {final_mae:.4f}")
print(f"MSE: {final_mse:.4f}")
print(f"RMSE: {final_rmse:.4f}")


# torch.save(model.state_dict(), 'tft_model.pth')
# # Load the model later
# model.load_state_dict(torch.load('tft_model.pth'))
