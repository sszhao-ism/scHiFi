from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from .data import ProcessedData, take_rows
from .graph import (
    GlobalGraphs,
    expand_halo,
    induced_subgraphs,
    scipy_to_torch_sparse,
)
from .metrics import evaluate_clustering
from .model import ScHiFi
from .partition import GraphPartitions, partition_global_graph


@dataclass
class TrainerConfig:
    device: str = "cuda"
    inference_device: str = "auto"
    train_mode: str = "auto"  # auto, full, cluster
    full_batch_threshold: int = 6000
    partition_method: str = "auto"
    cluster_size: int = 1024
    halo_hops: int = 1
    max_halo_nodes: Optional[int] = 4096
    pretrain_epochs: int = 400
    train_epochs: int = 150
    pretrain_lr: float = 1e-3
    train_lr: float = 0.5
    update_interval: int = 1
    tol: float = 1e-3
    attr_batch_size: int = 256
    cluster_weight: float = 0.5
    zinb_weight: float = 1.0
    graph_weight: float = 0.1
    align_weight: float = 0.1
    negative_ratio: float = 1.0
    max_positive_edges: int = 20000
    seed: int = 1111
    log_interval: int = 10


@dataclass
class TrainResult:
    labels: np.ndarray
    latent: np.ndarray
    metrics: Dict[str, float]
    history: List[Dict[str, float]] = field(default_factory=list)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _sample_graph_reconstruction_loss(
    embeddings: torch.Tensor,
    raw_adjacency: sp.csr_matrix,
    negative_ratio: float,
    max_positive_edges: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Sampled link reconstruction; never materializes an N x N matrix."""
    upper = sp.triu(raw_adjacency, k=1).tocoo()
    n_pos_available = int(upper.nnz)
    if n_pos_available == 0:
        return embeddings.sum() * 0.0
    n_pos = min(n_pos_available, int(max_positive_edges))
    if n_pos < n_pos_available:
        chosen = rng.choice(n_pos_available, size=n_pos, replace=False)
        pos_u = upper.row[chosen]
        pos_v = upper.col[chosen]
    else:
        pos_u, pos_v = upper.row, upper.col

    n_nodes = raw_adjacency.shape[0]
    n_neg = max(1, int(round(n_pos * negative_ratio)))
    neg_u: List[int] = []
    neg_v: List[int] = []
    attempts = 0
    max_attempts = max(1000, n_neg * 30)
    while len(neg_u) < n_neg and attempts < max_attempts:
        batch = min((n_neg - len(neg_u)) * 3, 65536)
        u = rng.integers(0, n_nodes, size=batch, endpoint=False)
        v = rng.integers(0, n_nodes, size=batch, endpoint=False)
        valid = u != v
        u, v = u[valid], v[valid]
        # Sparse advanced indexing returns a matrix; A1 converts to a flat array.
        absent = np.asarray(raw_adjacency[u, v]).reshape(-1) == 0
        for a, b in zip(u[absent], v[absent]):
            neg_u.append(int(a))
            neg_v.append(int(b))
            if len(neg_u) >= n_neg:
                break
        attempts += batch
    if len(neg_u) < n_neg:
        raise RuntimeError("Negative-edge sampling failed; graph may be unexpectedly dense.")

    device = embeddings.device
    pos_u_t = torch.as_tensor(pos_u, dtype=torch.long, device=device)
    pos_v_t = torch.as_tensor(pos_v, dtype=torch.long, device=device)
    neg_u_t = torch.as_tensor(neg_u, dtype=torch.long, device=device)
    neg_v_t = torch.as_tensor(neg_v, dtype=torch.long, device=device)
    pos_logits = (embeddings[pos_u_t] * embeddings[pos_v_t]).sum(dim=1)
    neg_logits = (embeddings[neg_u_t] * embeddings[neg_v_t]).sum(dim=1)
    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
    return F.binary_cross_entropy_with_logits(logits, labels)


class ScHiFiTrainer:
    def __init__(
        self,
        model: ScHiFi,
        data: ProcessedData,
        graphs: GlobalGraphs,
        config: TrainerConfig,
        output_dir: Union[str, Path],
    ):
        pass

    def _resolve_inference_device(self) -> torch.device:
        pass

    def _iter_core_parts(self, epoch: int) -> Iterable[np.ndarray]:
        pass

    def _prepare_subgraph(
        self, core_nodes: np.ndarray
    )

    def _batch_forward(
        self,
        core_nodes: np.ndarray,
        add_noise: bool,
    ):
        pass

    def _zinb_inputs(self, core_nodes: np.ndarray):
        raw = torch.from_numpy(take_rows(self.data.raw_counts, core_nodes)).to(self.device)
        sf = torch.from_numpy(self.data.size_factors[core_nodes]).to(self.device)
        return raw, sf

    def pretrain(self, checkpoint_path: Union[str, Path]) -> None:
        pass

    def load_pretrained(self, checkpoint_path: Union[str, Path]) -> None:
        pass

    @torch.no_grad()
    def _exact_topology_all(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact full-global-graph GCN inference, independent of training partitions."""
        pass

    @torch.no_grad()
    def encode_all(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @torch.no_grad()
    def _q_and_target_all(self) -> Tuple[np.ndarray, torch.Tensor]:
        latent, _ = self.encode_all()
        q_parts = []
        bs = self.config.attr_batch_size
        for start in range(0, latent.shape[0], bs):
            z = torch.from_numpy(latent[start : start + bs]).to(self.device)
            q_parts.append(self.model.soft_assign(z).cpu())
        q = torch.cat(q_parts)
        p = self.model.target_distribution(q)
        return latent, p

    def initialize_clustering(self) -> np.ndarray:
        latent, _ = self.encode_all()
        kmeans = KMeans(
            n_clusters=self.model.n_clusters,
            init="k-means++",
            n_init=20,
            random_state=self.config.seed,
        )
        labels = kmeans.fit_predict(latent)
        centers = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(self.device)
        self.model.initialize_cluster_centers(centers)
        print("Initialized cluster centers with k-means++ on exact global embeddings")
        return labels

    def fit(self) -> TrainResult:
        pass

    def _save_outputs(
        self,
        latent: np.ndarray,
        labels: np.ndarray,
        metrics: Dict[str, float],
    ) -> None:
        np.savetxt(self.output_dir / "latent.csv", latent, delimiter=",")
        np.savetxt(self.output_dir / "pred_labels.csv", labels, fmt="%d", delimiter=",")
        (self.output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        if self.history:
            pd.DataFrame(self.history).to_csv(self.output_dir / "history.csv", index=False)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "n_clusters": self.model.n_clusters,
            },
            self.output_dir / "final_model.pt",
        )
        print(f"Saved final outputs to {self.output_dir}")
