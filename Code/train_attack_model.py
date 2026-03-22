import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_attack_model(
    npz_path: str | None = None,
):
    if npz_path is None:
        npz_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results",
            "attack_dataset.npz",
        )

    data = np.load(npz_path)
    X_attack_train = data["X"]
    y_attack_train = data["y"]

    attack_model = RandomForestClassifier(n_estimators=100)
    attack_model.fit(X_attack_train, y_attack_train)
    train_preds = attack_model.predict(X_attack_train)
    train_acc = float(accuracy_score(y_attack_train, train_preds))

    return {
        "attack_model": attack_model,
        "train_accuracy": train_acc,
        "npz_path": npz_path,
    }


if __name__ == "__main__":
    result = train_attack_model()
    print(f"Attack Model Training Accuracy: {result['train_accuracy']:.4f}")
