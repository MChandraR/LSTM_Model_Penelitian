import torch

def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, 0])  
        y.append(data[i+seq_length, 1])    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)