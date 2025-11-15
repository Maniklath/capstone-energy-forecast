# src/train.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from src.data.preprocess import load_and_preprocess
from src.data.dataloader import get_dataloaders
from src.models.models import LSTMForecaster
import os

def train(args):
    X, y, scaler = load_and_preprocess(args.input, seq_len=args.seq_len)
    train_loader, val_loader = get_dataloaders(X, y, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMForecaster(input_dim=X.shape[2], hidden_dim=args.hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        print(f"Epoch {epoch} train_loss={train_loss:.4f}")
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/checkpoint.pth")
    print("Saved model to models/checkpoint.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample/sample_load.csv")
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    args = parser.parse_args()
    train(args)
