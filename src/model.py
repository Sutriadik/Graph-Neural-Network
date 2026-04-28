"""
GIN (Graph Isomorphism Network) untuk Prediksi Inhibisi HIV.

Referensi:
- "How Powerful are Graph Neural Networks?" (Xu et al., ICLR 2019)
- OGB Baseline: https://github.com/snap-stanford/ogb

Arsitektur:
    - AtomEncoder  : memetakan fitur atom (9-dim) ke embedding 300-dim
    - GINConvLayer : message passing berbasis MLP + BondEncoder
    - BatchNorm    : normalisasi tiap layer
    - GlobalMeanPool: readout tingkat graf
    - MLP Head     : klasifikasi biner (HIV inhibitor: ya/tidak)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


# ---------------------------------------------------------------------------
# GIN Convolution Layer
# ---------------------------------------------------------------------------

class GINConvLayer(MessagePassing):
    """
    Satu layer GIN dengan BondEncoder untuk edge attributes.

    Formula:
        h_v^(k) = MLP[ (1 + ε) · h_v^(k-1)  +  Σ_{u∈N(v)} ReLU(h_u + e_uv) ]

    Args:
        hidden_dim (int): Dimensi embedding hidden state.
    """

    def __init__(self, hidden_dim: int):
        super().__init__(aggr="add")  # sum aggregation

        # 2-layer MLP setelah aggregasi
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

        # Learnable epsilon
        self.eps = nn.Parameter(torch.zeros(1))

        # Bond (edge) encoder — memetakan 3-dim bond features ke hidden_dim
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # Encode edge attributes
        edge_emb = self.bond_encoder(edge_attr)

        # (1 + ε)·h + Σ messages, lalu MLP
        aggregated = self.propagate(edge_index, x=x, edge_attr=edge_emb)
        out = self.mlp((1 + self.eps) * x + aggregated)
        return out

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Pesan dari tetangga: ReLU(h_neighbor + edge_embedding)."""
        return F.relu(x_j + edge_attr)


# ---------------------------------------------------------------------------
# Full GIN Model
# ---------------------------------------------------------------------------

class GINMolHIV(nn.Module):
    """
    Graph Isomorphism Network untuk ogbg-molhiv.

    Pipeline:
        Molekul (graph) → AtomEncoder → [GINConv + BN + ReLU + Dropout] × L
                       → GlobalMeanPool → MLP → logit (skor HIV inhibisi)

    Args:
        hidden_dim  (int)  : Dimensi hidden layer. Default: 300 (OGB baseline).
        num_layers  (int)  : Jumlah GIN layer. Default: 5.
        dropout     (float): Dropout rate. Default: 0.5.
        num_tasks   (int)  : Jumlah output. Default: 1 (binary classification).
    """

    def __init__(
        self,
        hidden_dim: int = 300,
        num_layers: int = 5,
        dropout: float = 0.5,
        num_tasks: int = 1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_tasks = num_tasks

        # ── Encoder ──────────────────────────────────────────────────────
        # OGB AtomEncoder: maps 9-dim atom features → hidden_dim embedding
        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)

        # ── GIN Layers ───────────────────────────────────────────────────
        self.convs = nn.ModuleList(
            [GINConvLayer(hidden_dim) for _ in range(num_layers)]
        )
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )

        # ── Graph-Level Readout MLP ───────────────────────────────────────
        self.graph_pred_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, num_tasks),
        )

    def forward(self, batch_data) -> torch.Tensor:
        """
        Forward pass.

        Args:
            batch_data: PyG Batch object dengan atribut
                        x, edge_index, edge_attr, batch.

        Returns:
            Tensor (N, num_tasks): logit prediksi per graf.
        """
        x = batch_data.x
        edge_index = batch_data.edge_index
        edge_attr = batch_data.edge_attr
        batch = batch_data.batch

        # 1. Atom embedding
        h = self.atom_encoder(x)

        # 2. Message passing layers
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index, edge_attr)
            h = self.batch_norms[i](h)

            # ReLU + Dropout (kecuali layer terakhir)
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        # 3. Global graph readout (mean pooling atas semua node)
        h_graph = global_mean_pool(h, batch)

        # 4. Klasifikasi
        return self.graph_pred_mlp(h_graph)

    def get_node_embeddings(self, batch_data) -> torch.Tensor:
        """
        Kembalikan node-level embeddings setelah semua GIN layer.
        Berguna untuk visualisasi / explainability.
        """
        x = batch_data.x
        edge_index = batch_data.edge_index
        edge_attr = batch_data.edge_attr

        h = self.atom_encoder(x)
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index, edge_attr)
            h = self.batch_norms[i](h)
            if i < self.num_layers - 1:
                h = F.relu(h)
        return h

    def get_graph_embeddings(self, batch_data) -> torch.Tensor:
        """Kembalikan graph-level embeddings (sebelum classifier head)."""
        node_emb = self.get_node_embeddings(batch_data)
        return global_mean_pool(node_emb, batch_data.batch)

    def count_parameters(self) -> int:
        """Hitung total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("GINMolHIV — Model Summary")
    print("=" * 60)

    model = GINMolHIV(hidden_dim=300, num_layers=5, dropout=0.5)
    n_params = model.count_parameters()
    print(f"Total parameters : {n_params:,}")
    print(f"Hidden dim       : {model.hidden_dim}")
    print(f"GIN layers       : {model.num_layers}")
    print(f"Dropout          : {model.dropout}")
    print()
    print("Model architecture:")
    print(model)
