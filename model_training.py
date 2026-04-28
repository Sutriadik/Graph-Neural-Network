"""
model_training.py — GNN MolHIV Training Script
================================================

Dataset  : ogbg-molhiv (Open Graph Benchmark)
Model    : GIN (Graph Isomorphism Network)
Task     : Binary classification — apakah molekul menghambat HIV?
Metric   : ROC-AUC (standar OGB untuk ogbg-molhiv)

Cara menjalankan:
    python model_training.py

Output:
    - model/best_model.pt       : Model dengan val ROC-AUC tertinggi
    - model/final_model.pt      : Model pada epoch terakhir
    - model/training_history.pt : Dict berisi log training lengkap
    - assets/training_curves.png
    - assets/confusion_matrix.png
    - assets/roc_curve.png
    - assets/pr_curve.png
"""

import os
import time
import json
import functools
import torch
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data import Data
torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, Data])
import torch.nn as nn
import numpy as np

# PyTorch 2.6+ changed torch.load default to weights_only=True.
# OGB's internal loading code predates this and omits the argument, so patch it
# before OGB is imported so its torch.load calls continue to work.
torch.load = functools.partial(torch.load, weights_only=False)

from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from src.model import GINMolHIV
from src.utils import (
    evaluate_model,
    get_full_metrics,
    EarlyStopping,
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
)


# ─────────────────────────────────────────────────────────────────────────────
# Konfigurasi Hyperparameter
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # ── Dataset ────────────────────────────────────────────────────────────
    "dataset_name"   : "ogbg-molhiv",

    # ── Model ──────────────────────────────────────────────────────────────
    "hidden_dim"     : 300,     # Ukuran embedding — OGB baseline: 300
    "num_layers"     : 5,       # Jumlah GIN layer
    "dropout"        : 0.5,     # Dropout rate

    # ── Training ───────────────────────────────────────────────────────────
    "epochs"         : 100,     # Maksimum epoch
    "batch_size"     : 32,      # Batch size
    "lr"             : 1e-3,    # Learning rate (Adam)
    "weight_decay"   : 0.0,     # L2 regularization
    "num_workers"    : 0,       # DataLoader workers (0 = CPU)

    # ── Early Stopping ─────────────────────────────────────────────────────
    "patience"       : 20,      # Stop jika val AUC tidak meningkat
    "min_delta"      : 1e-4,

    # ── Paths ──────────────────────────────────────────────────────────────
    "model_dir"      : "model",
    "assets_dir"     : "assets",

    # ── Reproduksibilitas ──────────────────────────────────────────────────
    "seed"           : 42,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[Device] CPU (GPU tidak terdeteksi)")
    return device


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    """
    Satu epoch training.

    Returns:
        float: rata-rata training loss.
    """
    model.train()
    total_loss  = 0.0
    total_graphs = 0

    for batch in loader:
        batch   = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch)            # (B, 1)
        labels = batch.y.float()         # (B, 1)

        # Handle NaN labels
        valid_mask = ~torch.isnan(labels)
        if valid_mask.sum() == 0:
            continue

        loss = criterion(logits[valid_mask], labels[valid_mask])
        loss.backward()

        # Gradient clipping untuk stabilitas
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss   += loss.item() * valid_mask.sum().item()
        total_graphs += valid_mask.sum().item()

    return total_loss / max(total_graphs, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  GNN MolHIV — Training Script")
    print("  Dataset  : ogbg-molhiv (Open Graph Benchmark)")
    print("  Model    : Graph Isomorphism Network (GIN)")
    print("=" * 65)

    # ── Setup ────────────────────────────────────────────────────────────
    set_seed(CONFIG["seed"])
    device = get_device()
    os.makedirs(CONFIG["model_dir"],  exist_ok=True)
    os.makedirs(CONFIG["assets_dir"], exist_ok=True)

    # ── Load Dataset ─────────────────────────────────────────────────────
    print(f"\n[1/6] Mengunduh/memuat dataset '{CONFIG['dataset_name']}'...")
    dataset   = PygGraphPropPredDataset(name=CONFIG["dataset_name"])
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name=CONFIG["dataset_name"])

    train_dataset = dataset[split_idx["train"]]
    val_dataset   = dataset[split_idx["valid"]]
    test_dataset  = dataset[split_idx["test"]]

    print(f"  Total molekul  : {len(dataset):,}")
    print(f"  Train          : {len(train_dataset):,}")
    print(f"  Validation     : {len(val_dataset):,}")
    print(f"  Test           : {len(test_dataset):,}")
    print(f"  Node feat dim  : {dataset.num_node_features}")
    print(f"  Edge feat dim  : {dataset.num_edge_features}")

    # Statistik label
    labels = np.array([d.y.item() for d in train_dataset])
    pos = labels.sum()
    neg = len(labels) - pos
    print(f"\n  Label distribution (train):")
    print(f"    HIV Inhibitor (1): {int(pos):,}  ({100*pos/len(labels):.1f}%)")
    print(f"    Non-HIV       (0): {int(neg):,}  ({100*neg/len(labels):.1f}%)")
    print(f"    Imbalance ratio  : 1 : {neg/max(pos,1):.1f}")

    # ── DataLoader ───────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )

    # ── Build Model ──────────────────────────────────────────────────────
    print("\n[2/6] Membangun model GIN...")
    model = GINMolHIV(
        hidden_dim=CONFIG["hidden_dim"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
    ).to(device)

    print(f"  Total parameters: {model.count_parameters():,}")

    # ── Loss, Optimizer ──────────────────────────────────────────────────
    # Karena data imbalanced, gunakan pos_weight untuk BCE
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer  = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    # Learning rate scheduler (reduce on plateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10,
        min_lr=1e-5,
    )

    # ── Early Stopping ───────────────────────────────────────────────────
    early_stopping = EarlyStopping(
        patience=CONFIG["patience"],
        min_delta=CONFIG["min_delta"],
        mode="max",
        save_path=os.path.join(CONFIG["model_dir"], "best_model.pt"),
    )

    # ── Training Loop ────────────────────────────────────────────────────
    print(f"\n[3/6] Memulai training ({CONFIG['epochs']} epoch max)...")
    print("-" * 65)
    print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Val AUC':>9} | {'Test AUC':>9} | {'Time':>7}")
    print("-" * 65)

    history = {
        "train_loss": [], "val_auc": [], "test_auc": [],
        "best_val_auc": 0.0, "best_epoch": 0,
    }

    t_start = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        t_epoch = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate
        val_auc,  _, _ = evaluate_model(model, val_loader,  device)
        test_auc, _, _ = evaluate_model(model, test_loader, device)

        # Scheduler step
        scheduler.step(val_auc)

        # Log
        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_auc)
        history["test_auc"].append(test_auc)

        elapsed = time.time() - t_epoch
        print(
            f"{epoch:>6} | {train_loss:>11.4f} | {val_auc:>9.4f} | {test_auc:>9.4f} | {elapsed:>6.1f}s"
        )

        # Track best
        if val_auc > history["best_val_auc"]:
            history["best_val_auc"]  = val_auc
            history["best_epoch"]    = epoch
            history["best_test_auc"] = test_auc

        # Early stopping
        if early_stopping(val_auc, model):
            print(f"\n  [EarlyStopping] Berhenti di epoch {epoch}. Best val AUC: {early_stopping.best_value:.4f}")
            break

    total_time = time.time() - t_start
    print("-" * 65)
    print(f"\n  Total training time : {total_time/60:.1f} menit")
    print(f"  Best val  ROC-AUC  : {history['best_val_auc']:.4f}  (epoch {history['best_epoch']})")
    print(f"  Best test ROC-AUC  : {history.get('best_test_auc', 0.0):.4f}")

    # ── Simpan Final Model ───────────────────────────────────────────────
    final_model_path = os.path.join(CONFIG["model_dir"], "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"\n  Model final  disimpan: {final_model_path}")

    # Simpan history
    history_path = os.path.join(CONFIG["model_dir"], "training_history.pt")
    torch.save(history, history_path)
    torch.save(CONFIG,  os.path.join(CONFIG["model_dir"], "config.pt"))
    print(f"  History      disimpan: {history_path}")

    # ── Evaluasi Final (load best model) ─────────────────────────────────
    print("\n[4/6] Evaluasi final dengan best model...")
    best_model_path = os.path.join(CONFIG["model_dir"], "best_model.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_auc_final, y_true, y_pred = evaluate_model(model, test_loader, device)
    metrics = get_full_metrics(y_true, y_pred)

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  HASIL EVALUASI TEST SET                 │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  ROC-AUC   : {metrics['roc_auc']:.4f}                    │")
    print(f"  │  Accuracy  : {metrics['accuracy']:.4f}                    │")
    print(f"  │  Precision : {metrics['precision']:.4f}                    │")
    print(f"  │  Recall    : {metrics['recall']:.4f}                    │")
    print(f"  │  F1-Score  : {metrics['f1_score']:.4f}                    │")
    print(f"  │  Avg Prec  : {metrics['ap_score']:.4f}                    │")
    print(f"  └─────────────────────────────────────────┘")

    # Simpan metrics ke JSON
    with open(os.path.join(CONFIG["model_dir"], "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Visualisasi ──────────────────────────────────────────────────────
    print("\n[5/6] Menyimpan visualisasi...")

    plot_training_curves(
        history["train_loss"],
        history["val_auc"],
        history["test_auc"],
        save_path=os.path.join(CONFIG["assets_dir"], "training_curves.png"),
    )
    print("  assets/training_curves.png  ✓")

    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(CONFIG["assets_dir"], "confusion_matrix.png"),
    )
    print("  assets/confusion_matrix.png ✓")

    plot_roc_curve(
        y_true, y_pred,
        save_path=os.path.join(CONFIG["assets_dir"], "roc_curve.png"),
    )
    print("  assets/roc_curve.png        ✓")

    plot_pr_curve(
        y_true, y_pred,
        save_path=os.path.join(CONFIG["assets_dir"], "pr_curve.png"),
    )
    print("  assets/pr_curve.png         ✓")

    # ── Ringkasan ─────────────────────────────────────────────────────────
    print(f"\n[6/6] Selesai! ✓")
    print(f"\n  Model siap digunakan di app.py:")
    print(f"    streamlit run app.py")


if __name__ == "__main__":
    main()
