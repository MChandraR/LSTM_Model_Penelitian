from LSTM import LSTMModel
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import torch
import random
from util import create_sequences
import json


device = "cuda" if torch.cuda.is_available() else "cpu"

with open('config.json', 'r') as config:
    config = json.load(config)
    input_size = config["input_size"]
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    seq_length = config["seq_length"]

file_path = './dataset.xlsx'
data = pd.read_excel(file_path)
data_cleaned = data[['Kecepatan Angin', 'Tinggi Gelombang']]

scaler = MinMaxScaler(feature_range=(0, 1))

data_scaled = scaler.fit_transform(data_cleaned)
X, y = create_sequences(data_scaled, seq_length)
X = X.view(X.shape[0], X.shape[1], 1)

X = X.to(device)
y = y.to(device)

model = LSTMModel(input_size, hidden_size, num_layers, device).to(device)
#Lakukan training terlebih dahulu jika model belum ada
model.load_state_dict(torch.load('modelLSTM.pth'))
model.eval()

with torch.no_grad():
    predicted = model(X).cpu().detach().numpy()

predicted = scaler.inverse_transform(
    np.concatenate((X.cpu()[:, -1, 0].view(-1, 1).numpy(), predicted), axis=1)
)[:, 1]

for i in range(10):
    rand = random.randint(0, 50)
    print(f'Kecepatan Angin: {data_cleaned["Kecepatan Angin"].iloc[rand+seq_length]}, Prediksi Tinggi Gelombang: {predicted[rand]:.4f}')
