import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score


def plot_confusions(cms, title, path):
    fig, axes = plt.subplots(int(np.ceil(len(cms) / 3)), 3, figsize=(12.6, 7.6))
    axes = np.array(axes).reshape(-1)
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Non-recycler", "Recycler"], yticklabels=["Non-recycler", "Recycler"])
        ax.set_title(name)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
    for ax in axes[len(cms):]:
        ax.axis("off")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_roc(probs, y_test, title, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, prob in probs.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, prob):.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.500)")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
