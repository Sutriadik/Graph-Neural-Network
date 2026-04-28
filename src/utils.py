"""
Utility Functions — GNN MolHIV Project

Berisi:
    - mol_to_graph      : Konversi SMILES string → PyG Data object
    - evaluate_model    : Evaluasi ROC-AUC, precision, recall, F1
    - EarlyStopping     : Callback early stopping
    - plot_training_curves  : Visualisasi loss & AUC
    - plot_confusion_matrix : Visualisasi confusion matrix
    - plot_roc_curve        : Visualisasi ROC curve
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from torch_geometric.data import Data
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# SMILES → Graph
# ─────────────────────────────────────────────────────────────────────────────

def mol_to_graph(smiles: str):
    """
    Konversi SMILES string ke PyG Data object menggunakan
    RDKit + OGB featurization (atom & bond features yang kompatibel
    dengan ogbg-molhiv).

    Args:
        smiles (str): SMILES notation molekul.

    Returns:
        torch_geometric.data.Data | None
    """
    try:
        from rdkit import Chem
        from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # ── Node features (atom) ─────────────────────────────────────────
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(atom_to_feature_vector(atom))
        x = torch.tensor(atom_features, dtype=torch.long)

        # ── Edge features (bond) — bidirectional ────────────────────────
        if mol.GetNumBonds() == 0:
            # Molekul tanpa bond (atom tunggal)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr  = torch.zeros((0, 3), dtype=torch.long)
        else:
            edge_indices = []
            edge_attrs   = []
            for bond in mol.GetBonds():
                i    = bond.GetBeginAtomIdx()
                j    = bond.GetEndAtomIdx()
                feat = bond_to_feature_vector(bond)
                # Graf tidak berarah → tambah dua arah
                edge_indices += [[i, j], [j, i]]
                edge_attrs   += [feat, feat]

            edge_index = torch.tensor(edge_indices, dtype=torch.long).T.contiguous()
            edge_attr  = torch.tensor(edge_attrs,   dtype=torch.long)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    except Exception as e:
        print(f"[mol_to_graph] Error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Evaluasi
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, loader, device: str = "cpu"):
    """
    Evaluasi model pada satu DataLoader.

    Returns:
        auc    (float) : ROC-AUC score
        y_true (ndarray): ground truth labels
        y_pred (ndarray): predicted probabilities
    """
    model.eval()
    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            logits = model(batch)
            probs  = torch.sigmoid(logits)

            y_true_list.append(batch.y.cpu().numpy())
            y_pred_list.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true_list).flatten()
    y_pred = np.concatenate(y_pred_list).flatten()

    # Handle NaN (OGB convention: NaN = label tidak diketahui)
    valid = ~np.isnan(y_true)
    y_true, y_pred = y_true[valid], y_pred[valid]

    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = 0.0

    return auc, y_true, y_pred


def get_full_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Hitung semua metrik evaluasi.

    Returns:
        dict berisi roc_auc, accuracy, precision, recall, f1, ap_score
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "roc_auc"   : float(roc_auc_score(y_true, y_pred_proba)),
        "accuracy"  : float(report["accuracy"]),
        "precision" : float(report.get("1", {}).get("precision", 0.0)),
        "recall"    : float(report.get("1", {}).get("recall", 0.0)),
        "f1_score"  : float(report.get("1", {}).get("f1-score", 0.0)),
        "ap_score"  : float(average_precision_score(y_true, y_pred_proba)),
    }
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Monitor validation metric dan simpan model terbaik.

    Args:
        patience (int) : Jumlah epoch tanpa peningkatan sebelum berhenti.
        min_delta (float): Peningkatan minimum yang dianggap signifikan.
        mode (str)     : "max" untuk AUC/accuracy, "min" untuk loss.
        save_path (str): Path untuk menyimpan best model.
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        mode: str = "max",
        save_path: str = "model/best_model.pt",
    ):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.save_path  = save_path

        self.best_value = -np.inf if mode == "max" else np.inf
        self.counter    = 0
        self.early_stop = False

    def __call__(self, value: float, model) -> bool:
        """
        Returns:
            True jika training harus dihentikan.
        """
        improved = (
            (value > self.best_value + self.min_delta)
            if self.mode == "max"
            else (value < self.best_value - self.min_delta)
        )

        if improved:
            self.best_value = value
            self.counter    = 0
            # Simpan model terbaik
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


# ─────────────────────────────────────────────────────────────────────────────
# Plot Functions
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "red"   : "#E24B4A",
    "green" : "#1D9E75",
    "blue"  : "#378ADD",
    "amber" : "#EF9F27",
    "purple": "#7F77DD",
    "gray"  : "#888780",
}

PLOT_STYLE = {
    "figure.facecolor" : "white",
    "axes.facecolor"   : "white",
    "axes.grid"        : True,
    "grid.alpha"       : 0.25,
    "grid.linestyle"   : "--",
    "font.family"      : "DejaVu Sans",
    "font.size"        : 11,
}


def plot_training_curves(
    train_losses: list,
    val_aucs: list,
    test_aucs: list,
    save_path: str = None,
):
    """Visualisasi loss training dan AUC validation/test per epoch."""
    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        epochs = range(1, len(train_losses) + 1)

        # ── Loss curve ──────────────────────────────────────────────────
        axes[0].plot(epochs, train_losses, color=PALETTE["red"], lw=2, label="Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Binary Cross-Entropy Loss")
        axes[0].set_title("Training Loss Curve", fontweight="bold")
        axes[0].legend()

        # ── ROC-AUC curve ───────────────────────────────────────────────
        axes[1].plot(epochs, val_aucs,  color=PALETTE["green"], lw=2,      label="Validation ROC-AUC")
        axes[1].plot(epochs, test_aucs, color=PALETTE["blue"],  lw=2, ls="--", label="Test ROC-AUC")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("ROC-AUC Score")
        axes[1].set_title("ROC-AUC per Epoch", fontweight="bold")
        axes[1].set_ylim([0.45, 1.0])
        axes[1].axhline(y=0.5, color=PALETTE["gray"], ls=":", lw=1, label="Random baseline")
        axes[1].legend()

        plt.suptitle("GIN Training Progress — ogbg-molhiv", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    save_path: str = None,
):
    """Plot confusion matrix pada threshold tertentu."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Non-HIV (0)", "HIV Inhibitor (1)"],
            yticklabels=["Non-HIV (0)", "HIV Inhibitor (1)"],
            ax=ax,
            linewidths=0.5,
        )
        ax.set_ylabel("True Label",      fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_title(f"Confusion Matrix  (threshold = {threshold})", fontweight="bold")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: str = None,
):
    """Plot ROC curve dengan shaded AUC area."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_val      = roc_auc_score(y_true, y_pred_proba)

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot(fpr, tpr, color=PALETTE["blue"], lw=2.5, label=f"GIN  (AUC = {auc_val:.4f})")
        ax.fill_between(fpr, tpr, alpha=0.10, color=PALETTE["blue"])
        ax.plot([0, 1], [0, 1], color=PALETTE["gray"], ls="--", lw=1.2, label="Random baseline")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
        ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
        ax.set_title("ROC Curve — Test Set", fontweight="bold")
        ax.legend(loc="lower right")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_pr_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: str = None,
):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot(recall, precision, color=PALETTE["green"], lw=2.5, label=f"GIN  (AP = {ap:.4f})")
        ax.fill_between(recall, precision, alpha=0.10, color=PALETTE["green"])
        ax.set_xlabel("Recall",    fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision-Recall Curve — Test Set", fontweight="bold")
        ax.legend()
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
