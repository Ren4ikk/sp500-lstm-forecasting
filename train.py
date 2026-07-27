import torch
import torch.nn as nn
from model import LSTMModel
from data import get_loaders

CSV_PATH = "all_stocks_5yr.csv"
EPOCHS   = 50
LR       = 1e-3
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_RATIO = 0.8


def eval_loss(model, loader, criterion):
    model.eval()
    total = 0
    with torch.no_grad():
        for x, y in loader:
            total += criterion(model(x.to(DEVICE)), y.to(DEVICE)).item()
    return total / len(loader)


def train():
    train_loader, test_loader = get_loaders(CSV_PATH, train_ratio=TRAIN_RATIO)
    model     = LSTMModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x.to(DEVICE)), y.to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        test_loss = eval_loss(model, test_loader, criterion)
        scheduler.step(test_loss)
        print(f"Epoch {epoch:02d}/{EPOCHS} | train={train_loss:.6f} | test={test_loss:.6f} | lr={optimizer.param_groups[0]['lr']:.2e}")

    torch.save(model.state_dict(), "lstm_stock.pt")
    print("Saved → lstm_stock.pt")


if __name__ == "__main__":
    train()
