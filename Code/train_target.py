import os
from datetime import datetime

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset

from Code.models import create_model
from Code.train import train_model


def train_target(
    checkpoints_dir: str | None = None,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 0.001,
    noise_multiplier: float = 1.1,
    max_grad_norm: float = 1.0,
    delta: float = 1e-5,
):
    if checkpoints_dir is None:
        checkpoints_dir = os.path.join(os.path.dirname(__file__), "checkpoints")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train = torch.load(os.path.join(checkpoints_dir, "X_train_target.pt"))
    y_train = torch.load(os.path.join(checkpoints_dir, "y_train_target.pt"))

    train_loader_np = DataLoader(
        TensorDataset(X_train.float(), y_train.long()),
        batch_size=batch_size,
        shuffle=True,
    )
    input_dim = int(X_train.shape[1])

    model_np = create_model(input_dim)
    optimizer_np = torch.optim.Adam(model_np.parameters(), lr=lr)
    criterion_np = nn.CrossEntropyLoss()

    history_np = train_model(
        model_np,
        train_loader_np,
        optimizer_np,
        criterion_np,
        epochs=epochs,
        device=device,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_np_path = os.path.join(checkpoints_dir, f"target_np_{run_id}.pth")
    torch.save(model_np.state_dict(), model_np_path)

    train_loader_dp = DataLoader(
        TensorDataset(X_train.float(), y_train.long()),
        batch_size=batch_size,
        shuffle=True,
    )
    model_dp = create_model(input_dim)
    optimizer_dp = torch.optim.Adam(model_dp.parameters(), lr=lr)
    criterion_dp = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine()
    model_dp, optimizer_dp, train_loader_dp = privacy_engine.make_private(
        module=model_dp,
        optimizer=optimizer_dp,
        data_loader=train_loader_dp,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    history_dp = train_model(
        model_dp,
        train_loader_dp,
        optimizer_dp,
        criterion_dp,
        epochs=epochs,
        device=device,
        privacy_engine=privacy_engine,
        delta=delta,
    )
    model_dp_path = os.path.join(checkpoints_dir, f"target_dp_{run_id}.pth")
    torch.save(model_dp.state_dict(), model_dp_path)

    return {
        "target_np": {"model": model_np, "history": history_np, "model_path": model_np_path},
        "target_dp": {"model": model_dp, "history": history_dp, "model_path": model_dp_path},
    }


if __name__ == "__main__":
    print(train_target())
