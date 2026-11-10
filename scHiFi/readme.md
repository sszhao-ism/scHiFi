# scHiFi

## scHiFi: Progressive Structure–Attribute Fusion for scRNA-seq Cell Clustering

This repository provides the official project page for the manuscript:

> **scHiFi: Progressive Structure–Attribute Fusion for scRNA-seq Cell Clustering**

**scHiFi** is an unsupervised representation learning framework for single-cell RNA sequencing (scRNA-seq) clustering. It jointly models gene-expression attributes and cell–cell topology through multi-scale graph learning and progressively integrates structural information into the attribute representation. A cluster-level cross-view consistency objective further coordinates topology-aware and fused representations within a unified learning framework.

---

## Repository Status

This repository is currently released as a public project record for manuscript submission.

The complete reproducibility package will be made publicly available after the manuscript is formally published, including:

- full training and evaluation code;
- preprocessing scripts;
- multi-scale KNN graph construction;
- implementation of the scHiFi framework;
- pretrained model weights;
- clustering evaluation scripts;
- visualization and biological interpretation scripts;
- differential expression and marker-gene analysis;
- functional enrichment and PPI analysis;
- configuration files for reproducing the main experiments.

At the current stage, only the project description and manuscript-related information are provided.

---

## Method Overview

scHiFi consists of four main components:

### 1. ZINB-based Attribute Modeling

A ZINB-based expression encoder is employed to model the sparsity, overdispersion, and count characteristics of scRNA-seq data, producing hierarchical attribute representations while preserving biologically informative expression patterns.

### 2. Multi-scale Topology Learning

Multiple KNN graphs with different neighborhood sizes are constructed to capture complementary cell–cell relationships at different structural scales. The topology encoder learns multi-scale neighborhood representations that characterize both fine-grained and broader cell connectivity patterns.

### 3. Progressive Structure–Attribute Fusion

Instead of performing one-shot feature fusion, scHiFi progressively injects topology-derived representations into matched layers of the attribute encoder. This hierarchical fusion strategy enables structural information to interact with gene-expression features at multiple representation levels, facilitating more coordinated structure–attribute learning.

### 4. Cross-view Clustering Consistency

A cluster-level consistency objective aligns the predictions derived from topology-aware representations with the target distribution obtained from the fused representation. This encourages complementary views to converge toward a shared clustering structure and improves the robustness of cell representation learning.

---

## Framework

The overall learning pipeline of scHiFi can be summarized as:

> **Gene Expression Modeling → Multi-scale Topology Learning → Progressive Structure–Attribute Fusion → Cross-view Consistency → Cell Clustering**

The framework is optimized jointly in an end-to-end manner.

---

## Key Features

- **Structure–attribute joint learning** for integrating gene-expression and cell-topology information.
- **Multi-scale graph modeling** for capturing complementary neighborhood structures.
- **Progressive hierarchical fusion** instead of simple one-stage feature aggregation.
- **Cross-view clustering consistency** for coordinating topology and fused representations.
- **Unsupervised cell clustering** without requiring cell-type annotations during training.
- **Biological validation** through downstream analyses such as marker genes, differential expression, enrichment analysis, and PPI analysis.

---

## Data Availability

All datasets used in the manuscript are publicly available from **GEO**, **SRA**, **Figshare**, or the corresponding original repositories.

Detailed dataset names, accession numbers, preprocessing procedures, and references are provided in the manuscript.

---

## Code Availability

The complete source code and analysis pipeline will be released after formal publication of the manuscript.

The future release will include:

```text
scHiFi/
├── data/
├── preprocessing/
├── graph/
├── models/
├── clustering/
├── evaluation/
├── biological_analysis/
├── configs/
├── train.py
└── README.md
```

---

## Reproducibility

The released implementation will provide all necessary scripts and configuration files for reproducing:

- data preprocessing;
- multi-scale graph construction;
- model training;
- clustering evaluation;
- representation visualization;
- differential expression analysis;
- marker-gene validation;
- enrichment analysis;
- PPI analysis.

---

## Citation

If you find this work useful, please cite the paper after it becomes publicly available.

```bibtex
@article{zhou2026schifi,
  title   = {scHiFi: Progressive Structure--Attribute Fusion for scRNA-seq Cell Clustering},
  author  = {Zhou, Zhongyang and Zhao, Shangshang and Wu, Haomin and Huang, Zhen and Wang, Wei and Chen, Feiyu},
  journal = {Information Fusion},
  year    = {2026},
  note    = {Under review}
}
```

---

## Contact

For questions regarding this project, please open an issue in this repository.

---

## License

The license information will be provided together with the public release of the complete source code.
