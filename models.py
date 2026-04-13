from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import dropout_edge, dropout_node, to_undirected


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
        use_raw_cell: bool = True,
        use_kind: bool = True,
        raw_cell_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(cat_vocab_sizes) != 3:
            raise ValueError("cat_vocab_sizes must have length 3.")
        self.emb_family = nn.Embedding(cat_vocab_sizes[0], cat_emb_dim)
        self.emb_cell = nn.Embedding(cat_vocab_sizes[1], cat_emb_dim)
        self.emb_kind = nn.Embedding(cat_vocab_sizes[2], cat_emb_dim)
        self.num_dims = num_dims
        self.clip_log_numeric = clip_log_numeric
        self.use_raw_cell = use_raw_cell
        self.use_kind = use_kind
        self.raw_cell_dropout = raw_cell_dropout
        self.unk_index = 0
        self.out_dim = cat_emb_dim + num_dims
        if self.use_raw_cell:
            self.out_dim += cat_emb_dim
        if self.use_kind:
            self.out_dim += cat_emb_dim

    def _apply_raw_cell_mask(self, raw_ids: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (not self.use_raw_cell) or self.raw_cell_dropout <= 0.0:
            return raw_ids
        mask = torch.rand_like(raw_ids.float()) < self.raw_cell_dropout
        out = raw_ids.clone()
        out[mask] = self.unk_index
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) < 3 + self.num_dims:
            raise ValueError(f"Expected at least {3+self.num_dims} feature dims, got {x.size(1)}")

        c0 = x[:, 0].long().clamp(min=0, max=self.emb_family.num_embeddings - 1)
        c1 = x[:, 1].long().clamp(min=0, max=self.emb_cell.num_embeddings - 1)
        c2 = x[:, 2].long().clamp(min=0, max=self.emb_kind.num_embeddings - 1)
        n = x[:, 3 : 3 + self.num_dims].float()

        if self.clip_log_numeric:
            n = torch.log1p(torch.clamp(n, min=0.0))

        pieces = [self.emb_family(c0)]
        if self.use_raw_cell:
            pieces.append(self.emb_cell(self._apply_raw_cell_mask(c1)))
        if self.use_kind:
            pieces.append(self.emb_kind(c2))
        pieces.append(n)
        return torch.cat(pieces, dim=1)


def _pool_three_ways(h: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    if h.numel() == 0:
        hidden = 0 if h.ndim < 2 else h.size(-1)
        return torch.zeros((num_graphs, hidden * 3), device=batch.device, dtype=h.dtype)
    return torch.cat(
        [
            global_mean_pool(h, batch, size=num_graphs),
            global_max_pool(h, batch, size=num_graphs),
            global_add_pool(h, batch, size=num_graphs),
        ],
        dim=1,
    )


def _pool_two_ways(h: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    if h.numel() == 0:
        hidden = 0 if h.ndim < 2 else h.size(-1)
        return torch.zeros((num_graphs, hidden * 2), device=batch.device, dtype=h.dtype)
    return torch.cat(
        [
            global_mean_pool(h, batch, size=num_graphs),
            global_max_pool(h, batch, size=num_graphs),
        ],
        dim=1,
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
        num_graphs = int(getattr(data, "num_graphs", int(batch.max().item()) + 1))
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        g = _pool_three_ways(h=h, batch=batch, num_graphs=num_graphs)
        return self.cls(g).view(-1)


class GNN4GateLike(nn.Module):
    """
    Closer Bi-GCN approximation:
    forward and reverse directed graphs processed by paired GCNs.
    """

    def __init__(self, encoder: NodeFeatureEncoder, hidden: int = 128, layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.dropout = dropout
        self.fwd = nn.ModuleList()
        self.bwd = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = encoder.out_dim
        for _ in range(layers):
            self.fwd.append(GCNConv(in_dim, hidden))
            self.bwd.append(GCNConv(in_dim, hidden))
            self.norms.append(nn.BatchNorm1d(hidden * 2))
            in_dim = hidden * 2

        self.cls = nn.Sequential(
            nn.Linear(hidden * 6, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, data):
        x = self.encoder(data.x)
        edge_index = data.edge_index
        edge_rev = edge_index.flip(0)
        batch = data.batch
        num_graphs = int(getattr(data, "num_graphs", int(batch.max().item()) + 1))
        h = x
        for cf, cb, norm in zip(self.fwd, self.bwd, self.norms):
            hf = cf(h, edge_index)
            hb = cb(h, edge_rev)
            h = torch.cat([hf, hb], dim=1)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        g = _pool_three_ways(h=h, batch=batch, num_graphs=num_graphs)
        return self.cls(g).view(-1)


class TrojanSaintLike(nn.Module):
    """
    Closer TrojanSAINT approximation:
    grouped gate-type features, undirected GraphSAGE, and subgraph-style sampling noise.
    """

    def __init__(
        self,
        encoder: NodeFeatureEncoder,
        hidden: int = 128,
        layers: int = 3,
        dropout: float = 0.3,
        edge_drop: float = 0.2,
        node_drop: float = 0.0,
        use_undirected: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.dropout = dropout
        self.edge_drop = edge_drop
        self.node_drop = node_drop
        self.use_undirected = use_undirected
        self.convs = nn.ModuleList()
        in_dim = encoder.out_dim
        for _ in range(layers):
            self.convs.append(SAGEConv(in_dim, hidden, aggr="mean"))
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
        num_graphs = int(getattr(data, "num_graphs", int(batch.max().item()) + 1))

        if self.use_undirected:
            edge_index = to_undirected(edge_index, num_nodes=x.size(0))

        h = x
        batch_cur = batch
        edge_cur = edge_index

        if self.training and self.node_drop > 0.0:
            edge_cur, _, node_mask = dropout_node(
                edge_cur,
                p=self.node_drop,
                num_nodes=h.size(0),
                training=True,
                relabel_nodes=True,
            )
            h = h[node_mask]
            batch_cur = batch[node_mask]

        for conv in self.convs:
            ei = edge_cur
            if self.training and self.edge_drop > 0:
                ei = dropout_edge(edge_cur, p=self.edge_drop, training=True)[0]
            h = conv(h, ei)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        g = _pool_two_ways(h=h, batch=batch_cur, num_graphs=num_graphs)
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
        num_graphs = int(getattr(data, "num_graphs", int(batch.max().item()) + 1))
        h = x
        for i, gat in enumerate(self.gats):
            h = gat(h, edge_index)
            if i < len(self.gats) - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        g = _pool_two_ways(h=h, batch=batch, num_graphs=num_graphs)
        return self.cls(g).view(-1)


def build_model(
    name: str,
    encoder: NodeFeatureEncoder,
    hidden: int,
    layers: int,
    dropout: float,
    model_overrides: Optional[dict] = None,
):
    key = name.strip().lower()
    extra = model_overrides or {}
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
            edge_drop=float(extra.get("edge_drop", 0.15)),
            node_drop=float(extra.get("node_drop", 0.10)),
            use_undirected=bool(extra.get("use_undirected", True)),
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
