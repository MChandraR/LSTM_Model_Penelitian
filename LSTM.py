import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

file_path = './dataset.xlsx'
data = pd.read_excel(file_path)
data_cleaned = data[['Kecepatan Angin', 'Tinggi Gelombang']]

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data_cleaned)

device = "cuda" if torch.cuda.is_available() else "cpu"
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, 0])  
        y.append(data[i+seq_length, 1])    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

seq_length = 3
X, y = create_sequences(data_scaled, seq_length)
X = X.view(X.shape[0], X.shape[1], 1)

X = X.to(device)
y = y.to(device)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 50).to(device)
        c0 = torch.zeros(2, x.size(0), 50).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Parameter model
input_size = 1
hidden_size = 50
num_layers = 2
model = LSTMModel(input_size, hidden_size, num_layers).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 3000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y.view(-1, 1))
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), "modelLSTM.pth")
model.eval()
with torch.no_grad():
    predicted = model(X).cpu().detach().numpy()

predicted = scaler.inverse_transform(
    np.concatenate((X.cpu()[:, -1, 0].view(-1, 1).numpy(), predicted), axis=1)
)[:, 1]

for i in range(5):
    print(f'Kecepatan Angin: {data_cleaned["Kecepatan Angin"].iloc[i+seq_length]}, Prediksi Tinggi Gelombang: {predicted[i]:.4f}')
