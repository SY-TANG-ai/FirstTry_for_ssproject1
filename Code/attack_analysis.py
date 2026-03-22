import os

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from Code.models import create_model


def _latest_checkpoint(checkpoints_dir: str, prefix: str):
    candidates = [
        f
        for f in os.listdir(checkpoints_dir)
        if f.startswith(prefix) and f.endswith(".pth")
    ]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint with prefix '{prefix}'")
    candidates.sort()
    return os.path.join(checkpoints_dir, candidates[-1])


def _get_attack_features(model, data_loader, device: str):
    model.eval()
    model.to(device)
    all_probs = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            sorted_probs, _ = torch.sort(probs, descending=True, dim=1)
            all_probs.append(sorted_probs.cpu().numpy())
    return np.vstack(all_probs)


def run_attack_analysis(
    checkpoints_dir: str | None = None,
    batch_size: int = 64,
):
    if checkpoints_dir is None:
        checkpoints_dir = os.path.join(os.path.dirname(__file__), "checkpoints")

    X_shadow_train = torch.load(os.path.join(checkpoints_dir, "X_train_shadow.pt"))
    y_shadow_train = torch.load(os.path.join(checkpoints_dir, "y_train_shadow.pt"))
    X_shadow_test = torch.load(os.path.join(checkpoints_dir, "X_test_shadow.pt"))
    y_shadow_test = torch.load(os.path.join(checkpoints_dir, "y_test_shadow.pt"))

    input_dim = int(X_shadow_train.shape[1])
    shadow_model = create_model(input_dim)
    shadow_path = _latest_checkpoint(checkpoints_dir, "shadow_")
    shadow_model.load_state_dict(torch.load(shadow_path, map_location="cpu"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    shadow_train_loader = DataLoader(
        TensorDataset(X_shadow_train.float(), y_shadow_train.long()),
        batch_size=batch_size,
        shuffle=False,
    )
    shadow_test_loader = DataLoader(
        TensorDataset(X_shadow_test.float(), y_shadow_test.long()),
        batch_size=batch_size,
        shuffle=False,
    )

    mem_probs = _get_attack_features(shadow_model, shadow_train_loader, device)
    non_mem_probs = _get_attack_features(shadow_model, shadow_test_loader, device)

    X_attack = np.vstack([mem_probs, non_mem_probs])
    y_attack = np.concatenate(
        [np.ones(len(mem_probs)), np.zeros(len(non_mem_probs))]
    ).astype("int64")

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    npz_path = os.path.join(results_dir, "attack_dataset.npz")
    np.savez(npz_path, X=X_attack, y=y_attack)

    attack_model = RandomForestClassifier(n_estimators=100)
    attack_model.fit(X_attack, y_attack)
    train_preds = attack_model.predict(X_attack)
    train_acc = float(accuracy_score(y_attack, train_preds))

    return {
        "shadow_checkpoint": shadow_path,
        "attack_train_accuracy": train_acc,
        "attack_dataset_path": npz_path,
    }


if __name__ == "__main__":
    print(run_attack_analysis())
