import os
import sys

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Code.eval_utils import calculate_asr, plot_roc_curve
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


def _load_model(model_path, input_dim, device):
    model = create_model(input_dim)
    state = torch.load(model_path, map_location="cpu")
    if any(k.startswith("_module.") for k in state.keys()):
        state = {k.replace("_module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _extract_probabilities(model, data, device):
    inputs = torch.tensor(data, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        sorted_probs, _ = torch.sort(probs, descending=True, dim=1)
    return sorted_probs.cpu().numpy()


def run_evaluation(
    checkpoints_dir: str | None = None,
    results_dir: str | None = None,
):
    if checkpoints_dir is None:
        checkpoints_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

    attack_data = np.load(os.path.join(results_dir, "attack_dataset.npz"))
    X_attack_train = attack_data["X"]
    y_attack_train = attack_data["y"]

    attack_classifier = RandomForestClassifier(n_estimators=100)
    attack_classifier.fit(X_attack_train, y_attack_train)

    X_target_train = torch.load(os.path.join(checkpoints_dir, "X_train_target.pt")).numpy()
    X_target_test = torch.load(os.path.join(checkpoints_dir, "X_test_target.pt")).numpy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = int(X_target_train.shape[1])
    target_np_path = _latest_checkpoint(checkpoints_dir, "target_np_")
    target_dp_path = _latest_checkpoint(checkpoints_dir, "target_dp_")
    target_model_np = _load_model(target_np_path, input_dim, device)
    target_model_dp = _load_model(target_dp_path, input_dim, device)

    X_target_np_eval = np.vstack(
        [
            _extract_probabilities(target_model_np, X_target_train, device),
            _extract_probabilities(target_model_np, X_target_test, device),
        ]
    )
    y_target_eval = np.concatenate(
        [np.ones(len(X_target_train)), np.zeros(len(X_target_test))]
    )
    preds_np = attack_classifier.predict(X_target_np_eval)
    probs_np = attack_classifier.predict_proba(X_target_np_eval)[:, 1]
    asr_np = calculate_asr(y_target_eval, preds_np)
    roc_np_path = os.path.join(results_dir, "roc_np.png")
    auc_np = plot_roc_curve(y_target_eval, probs_np, "Non-Private Model", roc_np_path)

    X_target_dp_eval = np.vstack(
        [
            _extract_probabilities(target_model_dp, X_target_train, device),
            _extract_probabilities(target_model_dp, X_target_test, device),
        ]
    )
    preds_dp = attack_classifier.predict(X_target_dp_eval)
    probs_dp = attack_classifier.predict_proba(X_target_dp_eval)[:, 1]
    asr_dp = calculate_asr(y_target_eval, preds_dp)
    roc_dp_path = os.path.join(results_dir, "roc_dp.png")
    auc_dp = plot_roc_curve(y_target_eval, probs_dp, "DP-Private Model", roc_dp_path)

    return {
        "attack_train_accuracy": float(attack_classifier.score(X_attack_train, y_attack_train)),
        "asr_np": asr_np,
        "asr_dp": asr_dp,
        "auc_np": auc_np,
        "auc_dp": auc_dp,
        "roc_np_path": roc_np_path,
        "roc_dp_path": roc_dp_path,
    }


if __name__ == "__main__":
    print(run_evaluation())
