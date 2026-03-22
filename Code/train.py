from typing import Dict, Optional


def train_model(
    model,
    train_loader,
    optimizer,
    criterion,
    epochs: int = 10,
    device: Optional[str] = None,
    privacy_engine=None,
    delta: float = 1e-5,
) -> Dict[str, list]:
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.train()

    losses = []
    accs = []
    epsilons = []

    for _ in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * labels.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += int((preds == labels).sum().item())
            total_count += int(labels.size(0))

        if total_count:
            losses.append(total_loss / total_count)
            accs.append(total_correct / total_count)
        else:
            losses.append(0.0)
            accs.append(0.0)

        if privacy_engine is not None:
            epsilons.append(float(privacy_engine.get_epsilon(delta=delta)))
        else:
            epsilons.append(None)

    return {"losses": losses, "accs": accs, "epsilons": epsilons}
