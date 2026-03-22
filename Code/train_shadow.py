import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from Code.models import create_model
from Code.train import train_model


def train_shadow(
    checkpoints_dir: str | None = None,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 0.001,
    device: str | None = None,
):
    if checkpoints_dir is None:
        checkpoints_dir = os.path.join(os.path.dirname(__file__), "checkpoints")

    X_train = torch.load(os.path.join(checkpoints_dir, "X_train_shadow.pt"))
    y_train = torch.load(os.path.join(checkpoints_dir, "y_train_shadow.pt"))

    train_loader = DataLoader(
        TensorDataset(X_train.float(), y_train.long()),
        batch_size=batch_size,
        shuffle=True,
    )

    input_dim = int(X_train.shape[1])
    model = create_model(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = train_model(
        model,
        train_loader,
        optimizer,
        criterion,
        epochs=epochs,
        device=device,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_path = os.path.join(checkpoints_dir, f"shadow_{run_id}.pth")
    torch.save(model.state_dict(), model_path)

    return {"model": model, "history": history, "model_path": model_path}
