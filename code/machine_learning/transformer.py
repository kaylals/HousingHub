# pip install pytorch-forecasting



import pytorch_forecasting
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.examples import get_stallion_data
from pytorch_forecasting import TemporalFusionTransformer, Trainer
from pytorch_forecasting.metrics import SMAPE

# Prepare Data

# Define the training and validation datasets
max_encoder_length = input_steps
max_prediction_length = output_steps

# Create TimeSeriesDataSet
training = TimeSeriesDataSet(
    train,
    time_idx="time_idx",
    target="Log Price",
    group_ids=["group_id"],  # Modify this according to your data
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],
    static_reals=[],
    time_varying_known_categoricals=[],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["Log Price"],
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet(
    val,
    time_idx="time_idx",
    target="Log Price",
    group_ids=["group_id"],  # Modify this according to your data
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],
    static_reals=[],
    time_varying_known_categoricals=[],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["Log Price"],
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# Create DataLoaders

from pytorch_forecasting import TemporalFusionTransformer, Trainer
from pytorch_forecasting.data import NaNLabelEncoder

train_dataloader = TimeSeriesDataLoader(training, batch_size=32, shuffle=True, num_workers=0)
val_dataloader = TimeSeriesDataLoader(validation, batch_size=32, shuffle=False, num_workers=0)


# Initialize and Train the TFT Model

model = TemporalFusionTransformer(
    hidden_size=16,
    lstm_hidden_size=16,
    attention_head_size=4,
    dropout=0.1,
    output_size=output_steps,
    loss=SMAPE(),
    # other parameters can be adjusted
)

trainer = Trainer(
    max_epochs=50,
    gpus=1 if torch.cuda.is_available() else 0,
)

trainer.fit(model, train_dataloader, val_dataloader)


# Predict using the trained model
model.eval()
predictions = []
with torch.no_grad():
    for batch in val_dataloader:
        x = batch["encoder"]
        prediction = model(x)
        predictions.append(prediction)

predictions = torch.cat(predictions, dim=0).cpu().numpy()


# torch.save(model.state_dict(), 'tft_model.pth')
# # Load the model later
# model.load_state_dict(torch.load('tft_model.pth'))
