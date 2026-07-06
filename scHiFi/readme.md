# scHiFi

**scHiFi: Single-cell Hierarchical Fusion Network with Cross-View Consistency for scRNA-seq Clustering**

This repository provides the official project page for the manuscript:

> scHiFi: Single-cell Hierarchical Fusion Network with Cross-View Consistency for scRNA-seq Clustering

scHiFi is a hierarchical information fusion framework for unsupervised single-cell RNA sequencing (scRNA-seq) clustering. It integrates ZINB-based expression modeling, multi-scale cell topology encoding, hierarchical structure–attribute fusion, and cluster-level cross-view consistency into a unified optimization framework.

## Repository status

This repository is currently released as a public project record for manuscript submission.

The complete reproducibility package will be made publicly available after the manuscript is formally published, including:

- full training and evaluation code;
- preprocessing scripts;
- multi-scale graph construction modules;
- model implementation of scHiFi;
- pretrained model weights;
- scripts for clustering evaluation;
- scripts for visualization and biological interpretation;
- scripts for differential expression analysis, marker gene validation, enrichment analysis, and PPI analysis;
- configuration files for reproducing the main experiments.

At the current stage, only a minimal project entry file and manuscript-related information are provided.

## Method overview

scHiFi contains four main components:

1. **ZINB-based attribute encoder** for modeling sparse and overdispersed scRNA-seq count data.
2. **Multi-scale topology encoder** for extracting hierarchical cell-neighborhood representations from KNN graphs with different neighborhood sizes.
3. **Hierarchical progressive structure–attribute fusion** for injecting graph-derived topological embeddings into matched layers of the expression encoder.
4. **Cluster-level cross-view consistency** for aligning topology-view predictions with the target distribution derived from the fused representation.

## Data availability

All datasets used in the manuscript are publicly available from GEO, SRA, Figshare, or the corresponding original repositories. Dataset accession information is provided in the manuscript.

## Code availability

The full source code and analysis scripts will be released after formal publication of the manuscript.

## Citation

If you find this work useful, please cite our paper after it becomes available.

```bibtex
@article{zhou2026schifi,
  title   = {scHiFi: Single-cell Hierarchical Fusion Network with Cross-View Consistency for scRNA-seq Clustering},
  author  = {Zhou, Zhongyang and Zhao, Shangshang and Wu, Haomin and Huang, Zhen and Wang, Wei and Chen, Feiyu},
  journal = {Information Fusion},
  year    = {2026},
  note    = {Under review}
}