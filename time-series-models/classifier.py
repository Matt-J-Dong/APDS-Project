import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_probs=[0.1, 0.2, 0.3]):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, output_dim)
        self.dropout1 = nn.Dropout(p=dropout_probs[0])  # ~r 0-out neurons. Small dropout for shallow layers
        self.dropout2 = nn.Dropout(p=dropout_probs[1])  # Medium dropout deeper in the network
        self.dropout3 = nn.Dropout(p=dropout_probs[2])  # Slightly higher dropout in deeper layers

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.dropout3(x)
        logits = self.fc5(x)
        probs = torch.softmax(logits, dim=1) 
        return logits, probs

    def forward_mc(self, x, mc_samples=10):
        self.train()  # Ensure dropout is active
        preds = torch.stack([self.forward(x)[1] for _ in range(mc_samples)], dim=0)
        return preds

def train(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

# Ensure `train` is exportable
__all__ = ["MLP", "train"]