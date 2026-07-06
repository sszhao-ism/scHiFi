from __future__ import annotations

import argparse
from pathlib import Path

from schifi.data import load_dataset, preprocess_counts
from schifi.graph import build_or_load_global_graphs
from schifi.model import ScHiFi
from schifi.trainer import ScHiFiTrainer, TrainerConfig
from schifi.local_trainer import BatchLocalScHiFiTrainer, LocalTrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="scHiFi with global sparse KNN graphs and Cluster-GCN-style scalable training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="Quake_10x_Limb_Muscle")
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="./Dataset")
    parser.add_argument("--label_key", type=str, default="cell_type1")
    parser.add_argument("--n_clusters", type=int, default=0, help="0 infers K from available labels")
    parser.add_argument("--n_hvg", type=int, default=2000)
    parser.add_argument("--ceil_noninteger", action="store_true")

    parser.add_argument("--k_list", type=int, nargs=4, default=[10, 15, 20, 25])
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--metric", type=str, default="euclidean")
    parser.add_argument("--graph_cache_dir", type=str, default="./graph_cache")
    parser.add_argument("--rebuild_graph", action="store_true")

    parser.add_argument("--train_mode", choices=["auto", "full", "cluster", "local"], default="auto")
    parser.add_argument("--local_batch_size", type=int, default=0, help="Batch size for --train_mode local; 0 uses --attr_batch_size.")
    parser.add_argument("--full_batch_threshold", type=int, default=6000)
    parser.add_argument("--partition_method", choices=["auto", "metis", "kmeans"], default="auto")
    parser.add_argument("--cluster_size", type=int, default=1024)
    parser.add_argument("--halo_hops", type=int, default=1)
    parser.add_argument("--max_halo_nodes", type=int, default=4096, help="0 disables the cap")

    parser.add_argument("--encoder_dims", type=int, nargs=3, default=[512, 256, 128])
    parser.add_argument("--decoder_dims", type=int, nargs=2, default=[128, 256])
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--alpha", type=float, default=1.0)

    parser.add_argument("--pretrain_epochs", type=int, default=400)
    parser.add_argument("--maxiter", type=int, default=150)
    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
    parser.add_argument(
        "--lr_fit", "--lr", dest="lr_fit", type=float, default=0.5,
        help="Main clustering-stage learning rate. --lr is retained as an alias.",
    )
    parser.add_argument("--update_interval", type=int, default=1)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument(
        "--attr_batch_size", "--batch_size", dest="attr_batch_size", type=int, default=256,
        help="Attribute/ZINB mini-batch size; --batch_size is a legacy alias.",
    )

    parser.add_argument("--cluster_weight", type=float, default=0.5)
    parser.add_argument("--zinb_weight", type=float, default=1.0)
    parser.add_argument("--graph_weight", type=float, default=0.1)
    parser.add_argument("--align_weight", type=float, default=0.1)
    parser.add_argument("--negative_ratio", type=float, default=1.0)
    parser.add_argument("--max_positive_edges", type=int, default=20000)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--inference_device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--output_dir", "--save_dir", dest="output_dir", type=str, default=None)
    parser.add_argument("--pretrained", "--ae_weights", dest="pretrained", type=str, default=None)
    parser.add_argument(
        "--pretrained_output", "--ae_weight_file", dest="pretrained_output", type=str, default=None,
        help="Path used to save/load the pretraining checkpoint when --pretrained is not supplied.",
    )
    parser.add_argument("--skip_pretrain", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_file = Path(args.data_file) if args.data_file else Path(args.data_root) / args.dataset / "data.h5"
    output_dir = Path(args.output_dir) if args.output_dir else Path("results") / args.dataset
    pretrained_path = (
        Path(args.pretrained)
        if args.pretrained
        else Path(args.pretrained_output)
        if args.pretrained_output
        else Path("AE_weights") / f"{args.dataset}_sparse_pretrained.pt"
    )

    print(f"Loading dataset: {data_file}")
    loaded = load_dataset(data_file, label_key=args.label_key)
    data = preprocess_counts(
        loaded,
        n_hvg=args.n_hvg,
        scale_attributes=True,
        ceil_noninteger=args.ceil_noninteger,
    )
    if args.n_clusters > 0:
        n_clusters = args.n_clusters
    elif data.labels is not None:
        n_clusters = int(data.labels.max() + 1)
    else:
        raise ValueError("Specify --n_clusters when labels are unavailable.")
    print(
        f"Processed {data.n_cells:,} cells, {data.n_genes:,} genes, "
        f"{data.x_graph.shape[1]:,} HVGs, K={n_clusters}"
    )

    model = ScHiFi(
        input_dim=data.n_genes,
        graph_dim=data.x_graph.shape[1],
        zinb_dim=data.n_genes,
        n_clusters=n_clusters,
        z_dim=args.z_dim,
        encoder_dims=args.encoder_dims,
        decoder_dims=args.decoder_dims,
        sigma=args.sigma,
        alpha=args.alpha,
    )

    if args.train_mode == "local":
        if args.pretrained is None and args.pretrained_output is None:
            pretrained_path = Path("AE_weights") / f"{args.dataset}_local_pretrained.pt"
        local_config = LocalTrainerConfig(
            device=args.device,
            inference_device=args.inference_device,
            local_batch_size=args.local_batch_size if args.local_batch_size > 0 else args.attr_batch_size,
            k_list=tuple(args.k_list),
            pca_dim=args.pca_dim,
            metric=args.metric,
            pretrain_epochs=args.pretrain_epochs,
            train_epochs=args.maxiter,
            pretrain_lr=args.pretrain_lr,
            train_lr=args.lr_fit,
            update_interval=args.update_interval,
            tol=args.tol,
            cluster_weight=args.cluster_weight,
            zinb_weight=args.zinb_weight,
            graph_weight=args.graph_weight,
            align_weight=args.align_weight,
            seed=args.seed,
            log_interval=args.log_interval,
        )
        trainer = BatchLocalScHiFiTrainer(model, data, local_config, output_dir)
    else:
        graphs = build_or_load_global_graphs(
            data.x_graph,
            k_list=args.k_list,
            pca_dim=args.pca_dim,
            metric=args.metric,
            cache_dir=args.graph_cache_dir,
            cache_key=args.dataset,
            rebuild=args.rebuild_graph,
            seed=args.seed,
        )
        config = TrainerConfig(
            device=args.device,
            inference_device=args.inference_device,
            train_mode=args.train_mode,
            full_batch_threshold=args.full_batch_threshold,
            partition_method=args.partition_method,
            cluster_size=args.cluster_size,
            halo_hops=args.halo_hops,
            max_halo_nodes=None if args.max_halo_nodes <= 0 else args.max_halo_nodes,
            pretrain_epochs=args.pretrain_epochs,
            train_epochs=args.maxiter,
            pretrain_lr=args.pretrain_lr,
            train_lr=args.lr_fit,
            update_interval=args.update_interval,
            tol=args.tol,
            attr_batch_size=args.attr_batch_size,
            cluster_weight=args.cluster_weight,
            zinb_weight=args.zinb_weight,
            graph_weight=args.graph_weight,
            align_weight=args.align_weight,
            negative_ratio=args.negative_ratio,
            max_positive_edges=args.max_positive_edges,
            seed=args.seed,
            log_interval=args.log_interval,
        )
        trainer = ScHiFiTrainer(model, data, graphs, config, output_dir)

    if pretrained_path.exists():
        trainer.load_pretrained(pretrained_path)
    elif args.skip_pretrain:
        print("Warning: --skip_pretrain was set and no pretrained checkpoint was found.")
    else:
        trainer.pretrain(pretrained_path)

    result = trainer.fit()
    if result.metrics:
        print("Final metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in result.metrics.items()))


if __name__ == "__main__":
    main()
