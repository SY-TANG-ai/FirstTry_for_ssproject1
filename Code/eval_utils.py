from sklearn.metrics import accuracy_score, auc, roc_curve


def calculate_asr(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))


def plot_roc_curve(y_true, y_probs, model_name, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = float(auc(fpr, tpr))

    if save_path:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return roc_auc
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title(f"Receiver Operating Characteristic (ROC) for {model_name}")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(save_path)

    return roc_auc
