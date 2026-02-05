# -*- coding: utf-8 -*-
"""
DS PROJECT LSTM MODEL STOCK PREDICTION.py
Converted from Colab notebook to a standalone Python script
"""

# DATA DOWNLOADING
# Using yfinance to fetch NVDA historical stock data
import yfinance as yf
import pandas as pd

symbol = "NVDA"
df = yf.download(symbol, start="1999-01-01", end="2026-01-31")  # You can adjust end date
df.reset_index(inplace=True)
df.to_csv("NVDA.csv", index=False)
print("Downloaded NVDA historical data and saved to NVDA.csv")

# DATA LOADING AND PREPROCESSING
# If you already have CSV, you can comment the above block and uncomment below:
# file_path = "NVDA.csv"
# df = pd.read_csv(file_path)

#convert the date column into datetime format for understanding
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
#display the dataframe now
print(df.head())

#calculating the moving average with a window size of 10
df['Moving_Average_10'] = df['Close'].rolling(window=10).mean()

#display the dataframe with the new column
print(df.head(15)) #displaying 15 rows from 0-14

#the first 10 values are nan because we need 10 values before to calculate the moving average with the window of 10

"""DATA CLEANING AND DATA PREPROCESSING"""

#check for nan values
nan_counts = df.isna().sum()

#display nan values
print(nan_counts)

#filling nan column for a newly added column moving_average_10 using backward fill
df['Moving_Average_10'].fillna(method='bfill', inplace=True, limit=10)
print(df.head(11))

print(df)

# **BUILDING LSTM MODEL**
#scaling the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Open', 'Volume', 'Moving_Average_10']])
df[['Open_scaled', 'Volume_scaled', 'Moving_Average_10_scaled']] = scaled_data

#90 days in advance prediction
#model recognize and capture the patterns of 90 days

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#Define window size for sequences
window_size=90

# Create sequences for training, validation and testing

#creating sequence of 90 rows, and appending them into the list
# len(df)-windows size, if length of df is 1000 and window size is 90 so loop run from 0 to 909, in iteration[0] first 90 rows, in iteration[909] rows from 909 to 1000
# formula of sequence overlapping i:i+window size, in below example for sake of simplicity i take window size 3
# the overlapping is happening because 0:3, 0 is the starting index and 3 in the ending index
# in a iteration[0] we get [0,1,2]
# the second iteration starts and end as 1:4
# in index 1 we have number 1 and in index 2 we have number 2 and the second iteration runs from index 1 to 4
# we have values[1,2,3]

sequences = []
for i in range(len(df) - window_size):
    seq = df[['Open_scaled', 'Volume_scaled', 'Moving_Average_10_scaled']].values[i:i+window_size]
    sequences.append(seq)

#convert sequences to pytorch tensors
sequences = torch.tensor(sequences, dtype=torch.float32)
X = sequences[:, :-1] #input sequences
y = sequences[:, -1] #target sequences

#split the data into training, validation and testing
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

num_samples = len(sequences)
train_samples = int(train_ratio*num_samples)
val_samples = int(val_ratio*num_samples)
test_samples = num_samples - train_samples - val_samples

X_train, X_val, X_test = X[:train_samples], X[train_samples:train_samples + val_samples], X[train_samples + val_samples:]
y_train, y_val, y_test = y[:train_samples], y[train_samples:train_samples + val_samples], y[train_samples + val_samples:]

#reshaping target tensors to have correct shape(batch_size and output_size)
y_train = y_train.view(-1, 3)
y_val = y_val.view(-1, 3)
y_test = y_test.view(-1, 3)

#creating dataloader for training, validation, and testing set with batch size
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#defining enhanced LSTM MODEL
class EnhancedLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size):
      super(EnhancedLSTM,self).__init__()
      self.hidden_size=hidden_size
      self.num_layers=num_layers
      self.lstm=nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
      self.fc=nn.Linear(hidden_size, output_size)
#Imagine an Analyst Reviewing Stock Data:
#The analyst reviews stock prices, trading volumes, and moving averages over 90 days.
#Each day's data contributes to their understanding of the overall trend.
#At the end of the 90 days, the analyst summarizes their findings and makes a prediction for the next few days.

#In this analogy:
#The LSTM is like the analyst, learning patterns day by day.
#The hidden state at the last timestep (out[:, -1, :]) is the analyst's summary.
#The FC layer is the decision-making process, converting the summary into actionable predictions.

  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
    out, _ = self.lstm(x,(h0,c0))
    #90th day, summmrizies the understanding of past 89 days and 90 day understanding is the conclusion of that
    out = self.fc(out[:, -1, :]) #get the output from the last time step only
    return out

# define hyperparameters and instantiate the model
input_size = 3  # Adjust based on your feature dimension
hidden_size = 64  # Best hyperparameter from your GridSearchCV
num_layers = 2  # Best hyperparameter from your GridSearchCV
output_size = 3  # Adjust based on your output dimension
model = EnhancedLSTM(input_size, hidden_size, num_layers, output_size)

# define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

#training loop with dataloader and validation
num_epochs = 100
train_losses=[]
val_losses=[]
best_val_loss = float('inf')
for epoch in range(num_epochs):
  model.train()

  for inputs, targets in train_loader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

  #Validation
  model.eval()
  val_loss = 0.0
  with torch.no_grad():
    for val_inputs, val_targets in val_loader:
      val_outputs = model(val_inputs)
      val_loss += criterion(val_outputs, val_targets)
  val_loss /= len(val_loader)
  val_losses.append(val_loss.item())

  print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}')

  # saving the model with best validation loss
  if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), 'best_model.pth')

# Plotting training loss over time
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting validation loss over time
plt.figure(figsize=(10, 6))
plt.plot(val_losses, label='Validation Loss', color='green')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Load the best model for testing
best_model = EnhancedLSTM(input_size, hidden_size, num_layers, output_size)
best_model.load_state_dict(torch.load('best_model.pth'))

# Evaluate model on testing set
best_model.eval()
test_loss = 0.0
with torch.no_grad():
    for test_inputs, test_targets in test_loader:
        test_outputs = best_model(test_inputs)
        test_loss += criterion(test_outputs, test_targets).item() * len(test_inputs)
test_loss /= len(test_loader.dataset)
print(f'Test Loss: {test_loss}')

# PREDICTION AND PLOTTING
#he prediction code is essentially asking the model:
#"Can you correctly predict the next value for the sequences from the data you were trained on?"
#It isn't predicting the future, just checking if the model learned past patterns correctly.
#it takes the actual data from df and plot it as a blue line and then takes the predicted value which model gives and see how well it predicts it

# Scale data again for prediction
scaled_data = scaler.fit_transform(df[['Open', 'Volume', 'Moving_Average_10']])
df[['Open_scaled', 'Volume_scaled', 'Moving_Average_10_scaled']] = scaled_data

# Create sequences for prediction
sequences_pred = []
for i in range(len(df) - window_size + 1):
    seq = df[['Open_scaled', 'Volume_scaled', 'Moving_Average_10_scaled']].values[i:i+window_size]
    sequences_pred.append(seq)

# Convert sequences to PyTorch tensor
X_pred = torch.tensor(sequences_pred, dtype=torch.float32)

# Set the model to evaluation mode
model.eval()

# Make predictions
with torch.no_grad():
    predicted_values = model(X_pred)

# Inverse scale predicted values
predicted_prices = scaler.inverse_transform(predicted_values.view(-1, 3))

# Create a new DataFrame for easier plotting
predicted_df = pd.DataFrame(predicted_prices, columns=['Predicted Open', 'Predicted Volume', 'Predicted Moving Average'])
predicted_df['Date'] = df['Date'][-len(predicted_prices):].reset_index(drop=True)

# Plotting

# Plot predicted Open prices
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Open'], label='Actual Open Price', color='blue')
plt.plot(predicted_df['Date'], predicted_df['Predicted Open'], label='Predicted Open Price', color='green')
plt.xlabel('Date')
plt.ylabel('Open Price (USD)')
plt.title('Actual vs Predicted Open Prices')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot predicted Volume
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Volume'], label='Actual Volume', color='blue')
plt.plot(predicted_df['Date'], predicted_df['Predicted Volume'], label='Predicted Volume', color='green')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('Actual vs Predicted Volume')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot predicted Moving Average
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Moving_Average_10'], label='Actual Moving Average', color='blue')
plt.plot(predicted_df['Date'], predicted_df['Predicted Moving Average'], label='Predicted Moving Average', color='green')
plt.xlabel('Date')
plt.ylabel('Moving Average')
plt.title('Actual vs Predicted Moving Average')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# **future data prediction**

# Get the last 90 data points for future prediction
last_90_data = df[['Open_scaled', 'Volume_scaled', 'Moving_Average_10_scaled']].tail(window_size).values
next_sequence_tensor = torch.tensor(last_90_data.reshape(1, window_size, 3), dtype=torch.float32)

# Predict the next 90 days
next_predictions = []
next_sequence = next_sequence_tensor.clone()
for _ in range(90):
    with torch.no_grad():
        next_prediction = model(next_sequence)
    next_predictions.append(next_prediction)
    next_sequence = torch.cat((next_sequence[:, 1:, :], next_prediction.unsqueeze(1)), dim=1)

# Inverse scale the predicted future values
next_prediction_prices = scaler.inverse_transform(torch.cat(next_predictions, dim=1).view(-1, 3))
next_prediction_dates = pd.date_range(df['Date'].max() + pd.DateOffset(days=1), periods=90, freq='D')
next_predicted_df = pd.DataFrame(next_prediction_prices, columns=['Predicted Open', 'Predicted Volume', 'Predicted Moving Average'])
next_predicted_df['Date'] = next_prediction_dates
print("Predicted future prices for next 90 days:")
print(next_predicted_df.head())

predicted_df = pd.concat([predicted_df, next_predicted_df], ignore_index=True)

# Plot future Open prices
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Open'], label='Actual Open Price', color='blue', linewidth=2)
plt.plot(predicted_df['Date'], predicted_df['Predicted Open'], label='Predicted Open Price', color='green', linestyle='--', linewidth=2)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Open Price (USD)', fontsize=12)
plt.title('Actual vs Predicted Open Prices for the Next 90 Days', fontsize=14)
plt.legend(fontsize=10)
plt.xticks(rotation=45, fontsize=10)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()
