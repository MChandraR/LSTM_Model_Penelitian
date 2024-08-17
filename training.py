import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from LSTM import LSTMModel
from util import create_sequences
import json

with open('config.json', 'r') as config:
    config = json.load(config)
    input_size = config["input_size"]
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    seq_length = config["seq_length"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]

file_path = './dataset.xlsx'
data = pd.read_excel(file_path)
data_cleaned = data[['Kecepatan Angin', 'Tinggi Gelombang']]

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data_cleaned)

#Cek device yang digunakan tergantung pada versi pytorch yang digunakan apakah cuda atau cpu
device = "cuda" if torch.cuda.is_available() else "cpu"

seq_length = 3
X, y = create_sequences(data_scaled, seq_length)
X = X.view(X.shape[0], X.shape[1], 1)

X = X.to(device)
y = y.to(device)

model = LSTMModel(input_size, hidden_size, num_layers, device).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y.view(-1, 1))
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
print("Training selesai... ")
torch.save(model.state_dict(), "modelLSTM.pth")
