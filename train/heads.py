from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from timegnn.models.gat_basic import DualGATModel
from timegnn.models.gat_time_decay import DualGATTimeAwareModel

from core.constants import NUM_EDGE_TYPES, NUM_PROMOTION_SLOTS


@dataclass
class ChessGATConfig:
    num_embedding_features: int = 15
    embedding_dims: int = 32
    gat_hidden_dim_event: int = 32
    gat_hidden_dim_embed: int = 32
    gat_hidden_dim_concat: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    use_batch_norm: bool = True
    activation: str = "elu"
    lambda_decay: float = 0.05
    policy_hidden_dim: int = 128
    num_mate_classes: int = 3


def _node_embedding_dim(cfg: ChessGATConfig) -> int:
    return cfg.gat_hidden_dim_concat * cfg.num_heads


class _NodeEmbeddingBackbone(nn.Module):
    def __init__(self, base_model: nn.Module, embed_dim: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.base_model.fc = nn.Identity()
        self.embed_dim = embed_dim

    def forward(self, data_event, **kwargs):
        return self.base_model(data_event, **kwargs)


def build_basic_backbone(cfg: ChessGATConfig, num_event_features: int) -> _NodeEmbeddingBackbone:
    embed_dim = _node_embedding_dim(cfg)
    model = DualGATModel(
        num_event_features=num_event_features,
        num_embedding_features=cfg.num_embedding_features,
        embedding_dims=cfg.embedding_dims,
        gat_hidden_dim_event=cfg.gat_hidden_dim_event,
        gat_hidden_dim_embed=cfg.gat_hidden_dim_embed,
        gat_hidden_dim_concat=cfg.gat_hidden_dim_concat,
        output_dim=embed_dim,
        num_heads=cfg.num_heads,
        edge_dim=NUM_EDGE_TYPES,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        use_batch_norm=cfg.use_batch_norm,
        activation=cfg.activation,
    )
    return _NodeEmbeddingBackbone(model, embed_dim)


def build_time_decay_backbone(cfg: ChessGATConfig, num_event_features: int) -> _NodeEmbeddingBackbone:
    embed_dim = _node_embedding_dim(cfg)
    model = DualGATTimeAwareModel(
        num_event_features=num_event_features,
        num_embedding_features=cfg.num_embedding_features,
        embedding_dims=cfg.embedding_dims,
        gat_hidden_dim_event=cfg.gat_hidden_dim_event,
        gat_hidden_dim_embed=cfg.gat_hidden_dim_embed,
        gat_hidden_dim_concat=cfg.gat_hidden_dim_concat,
        output_dim=embed_dim,
        num_heads=cfg.num_heads,
        lambda_decay=cfg.lambda_decay,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        use_batch_norm=cfg.use_batch_norm,
        activation=cfg.activation,
    )
    return _NodeEmbeddingBackbone(model, embed_dim)


class ValueHead(nn.Module):
    def __init__(self, embed_dim: int, num_mate_classes: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_mate_classes),
        )

    def forward(self, node_embeddings: torch.Tensor, batch_vector: torch.Tensor) -> torch.Tensor:
        graph_embed = global_mean_pool(node_embeddings, batch_vector)
        return self.proj(graph_embed)


class PolicyHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 2 + NUM_PROMOTION_SLOTS, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _decode_moves_vectorized(moves: torch.Tensor):
        promo_slot = moves % NUM_PROMOTION_SLOTS
        base = moves // NUM_PROMOTION_SLOTS
        from_sq = base // 64
        to_sq = base % 64
        return from_sq, to_sq, promo_slot

    def forward(
        self,
        node_embeddings: torch.Tensor,
        node_offsets: List[int],
        legal_move_indices: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        device = node_embeddings.device

        all_from = []
        all_to = []
        all_promo_slot = []
        graph_sizes = []

        for offset, moves in zip(node_offsets, legal_move_indices):
            if moves.numel() == 0:
                graph_sizes.append(0)
                continue
            from_sq, to_sq, promo_slot = self._decode_moves_vectorized(moves)
            all_from.append(from_sq + offset)
            all_to.append(to_sq + offset)
            all_promo_slot.append(promo_slot)
            graph_sizes.append(moves.numel())

        if not all_from:
            return [torch.zeros(0, device=device) for _ in node_offsets]

        from_sq_cat = torch.cat(all_from).to(device)
        to_sq_cat = torch.cat(all_to).to(device)
        promo_slot_cat = torch.cat(all_promo_slot).to(device)
        promo_onehot = F.one_hot(promo_slot_cat, num_classes=NUM_PROMOTION_SLOTS).float()

        pair = torch.cat(
            [node_embeddings[from_sq_cat], node_embeddings[to_sq_cat], promo_onehot], dim=-1
        )
        logits_flat = self.scorer(pair).squeeze(-1)

        logits_per_graph = []
        cursor = 0
        for size in graph_sizes:
            if size == 0:
                logits_per_graph.append(torch.zeros(0, device=device))
            else:
                logits_per_graph.append(logits_flat[cursor:cursor + size])
                cursor += size
        return logits_per_graph


class ChessDualHeadModel(nn.Module):
    def __init__(self, backbone: _NodeEmbeddingBackbone, cfg: ChessGATConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.value_head = ValueHead(backbone.embed_dim, cfg.num_mate_classes)
        self.policy_head = PolicyHead(backbone.embed_dim, cfg.policy_hidden_dim)

    def forward(self, data_event, node_offsets: List[int], legal_move_indices: List[torch.Tensor]):
        node_embeddings = self.backbone(data_event)
        value_logits = self.value_head(node_embeddings, data_event.batch)
        policy_logits = self.policy_head(node_embeddings, node_offsets, legal_move_indices)
        return policy_logits, value_logits