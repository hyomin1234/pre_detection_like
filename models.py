from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import dropout_edge


class NodeFeatureEncoder(nn.Module):
    """
    Encodes raw node feature vector:
    [family_id, cell_id, kind_id, 8 numeric fields]
    """

    def __init__(
        self,
        cat_vocab_sizes: List[int],
        cat_emb_dim: int = 16,
        num_dims: int = 8,
        clip_log_numeric: bool = True,
    ) -> None:
        super().__init__()
        if len(cat_vocab_sizes) != 3:
            raise ValueError("cat_vocab_sizes must have length 3.")
        self.emb_family = nn.Embedding(cat_vocab_sizes[0], cat_emb_dim)
        self.emb_cell = nn.Embedding(cat_vocab_sizes[1], cat_emb_dim)
        self.emb_kind = nn.Embedding(cat_vocab_sizes[2], cat_emb_dim)
        self.num_dims = num_dims
        self.clip_log_numeric = clip_log_numeric
        self.out_dim = cat_emb_dim * 3 + num_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) < 3 + self.num_dims:
            raise ValueError(f"Expected at least {3+self.num_dims} feature dims, got {x.size(1)}")

        c0 = x[:, 0].long().clamp(min=0, max=self.emb_family.num_embeddings - 1)
        c1 = x[:, 1].long().clamp(min=0, max=self.emb_cell.num_embeddings - 1)
        c2 = x[:, 2].long().clamp(min=0, max=self.emb_kind.num_embeddings - 1)
        n = x[:, 3 : 3 + self.num_dims].float()

        if self.clip_log_numeric:
            n = torch.log1p(torch.clamp(n, min=0.0))

        return torch.cat(
            [self.emb_family(c0), self.emb_cell(c1), self.emb_kind(c2), n], dim=1
        )


class RawSAGE(nn.Module):
    def __init__(self, encoder: NodeFeatureEncoder, hidden: int = 128, layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.dropout = dropout
        self.convs = nn.ModuleList()
        in_dim = encoder.out_dim
        for _ in range(layers):
            self.convs.append(SAGEConv(in_dim, hidden))
            in_dim = hidden
        self.cls = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, data):
        x = self.encoder(data.x)
        edge_index = data.edge_index
        batch = data.batch
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        g = torch.cat(
            [global_mean_pool(h, batch), global_max_pool(h, batch), global_add_pool(h, batch)],
            dim=1,
        )
        return self.cls(g).view(-1)


class GNN4GateLike(nn.Module):
    """
    Bidirectional GraphSAGE approximation:
    forward edges + reversed edges processed jointly per layer.
    """

    def __init__(self, encoder: NodeFeatureEncoder, hidden: int = 128, layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.dropout = dropout
        self.fwd = nn.ModuleList()
        self.bwd = nn.ModuleList()
        self.fuse = nn.ModuleList()

        in_dim = encoder.out_dim
        for _ in range(layers):
            self.fwd.append(SAGEConv(in_dim, hidden))
            self.bwd.append(SAGEConv(in_dim, hidden))
            self.fuse.append(nn.Linear(hidden * 2, hidden))
            in_dim = hidden

        self.cls = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, data):
        x = self.encoder(data.x)
        edge_index = data.edge_index
        edge_rev = edge_index.flip(0)
        batch = data.batch
        h = x
        for cf, cb, fuse in zip(self.fwd, self.bwd, self.fuse):
            hf = cf(h, edge_index)
            hb = cb(h, edge_rev)
            h = fuse(torch.cat([hf, hb], dim=1))
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        g = torch.cat(
            [global_mean_pool(h, batch), global_max_pool(h, batch), global_add_pool(h, batch)],
            dim=1,
        )
        return self.cls(g).view(-1)


class TrojanSaintLike(nn.Module):
    """
    Sampling-robust GraphSAGE approximation with stochastic edge dropout.
    """

    def __init__(
        self,
        encoder: NodeFeatureEncoder,
        hidden: int = 128,
        layers: int = 3,
        dropout: float = 0.3,
        edge_drop: float = 0.2,
    ):
        super().__init__()
        self.encoder = encoder
        self.dropout = dropout
        self.edge_drop = edge_drop
        self.convs = nn.ModuleList()
        in_dim = encoder.out_dim
        for _ in range(layers):
            self.convs.append(SAGEConv(in_dim, hidden))
            in_dim = hidden
        self.cls = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, data):
        x = self.encoder(data.x)
        edge_index = data.edge_index
        batch = data.batch
        h = x
        for conv in self.convs:
            ei = edge_index
            if self.training and self.edge_drop > 0:
                ei = dropout_edge(edge_index, p=self.edge_drop, training=True)[0]
            h = conv(h, ei)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        g = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)
        return self.cls(g).view(-1)


class FPGNNLike(nn.Module):
    """
    FP-GNN style attention baseline with multi-head GAT layers.
    """

    def __init__(
        self,
        encoder: NodeFeatureEncoder,
        hidden: int = 96,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.encoder = encoder
        self.dropout = dropout
        self.layers = layers

        self.gats = nn.ModuleList()
        in_dim = encoder.out_dim
        for _ in range(layers - 1):
            self.gats.append(GATConv(in_dim, hidden, heads=heads, dropout=dropout))
            in_dim = hidden * heads
        self.gats.append(GATConv(in_dim, hidden, heads=1, dropout=dropout))

        self.cls = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, data):
        x = self.encoder(data.x)
        edge_index = data.edge_index
        batch = data.batch
        h = x
        for i, gat in enumerate(self.gats):
            h = gat(h, edge_index)
            if i < len(self.gats) - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        g = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)
        return self.cls(g).view(-1)


def build_model(name: str, encoder: NodeFeatureEncoder, hidden: int, layers: int, dropout: float):
    key = name.strip().lower()
    if key == "raw_sage":
        return RawSAGE(encoder=encoder, hidden=hidden, layers=layers, dropout=dropout)
    if key == "gnn4gate_like":
        return GNN4GateLike(encoder=encoder, hidden=hidden, layers=layers, dropout=dropout)
    if key == "trojansaint_like":
        return TrojanSaintLike(
            encoder=encoder,
            hidden=hidden,
            layers=layers,
            dropout=max(dropout, 0.25),
            edge_drop=0.2,
        )
    if key == "fpgnn_like":
        return FPGNNLike(
            encoder=encoder,
            hidden=max(64, hidden // 2),
            heads=4,
            layers=max(2, layers),
            dropout=max(dropout, 0.2),
        )
    raise ValueError(f"Unknown model name: {name}")

