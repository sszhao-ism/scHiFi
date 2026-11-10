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
- multi-scale graph construction modules;
- model implementation of scHiFi;
- pretrained model weights;
- clustering evaluation scripts;
- visualization and biological interpretation scripts;
- differential expression analysis;
- marker gene validation;
- enrichment analysis and PPI analysis;
- configuration files for reproducing the main experiments.

At the current stage, only manuscript-related information and a minimal project record are provided.

---

## Method Overview

scHiFi consists of four main components:

### 1. ZINB-based Attribute Modeling

A ZINB-based expression encoder models the sparsity, overdispersion, and count characteristics of scRNA-seq data, producing informative attribute representations while preserving biologically relevant expression patterns.

### 2. Multi-scale Topology Learning

Multiple KNN graphs with different neighborhood sizes are constructed to capture complementary cell–cell relationships at different structural scales. The topology encoder learns hierarchical neighborhood representations from these graph views.

### 3. Progressive Structure–Attribute Fusion

Instead of performing one-stage feature fusion, scHiFi progressively injects topology-derived representations into matched layers of the attribute encoder. This strategy enables structural and expression information to interact across multiple representation levels.

### 4. Cross-view Clustering Consistency

A cluster-level consistency objective aligns topology-view predictions with the target distribution derived from the fused representation, encouraging different views to converge toward a shared clustering structure.

---

## Framework

The overall learning pipeline of scHiFi can be summarized as:

> **Gene Expression Modeling → Multi-scale Topology Learning → Progressive Structure–Attribute Fusion → Cross-view Consistency → Cell Clustering**

The entire framework is jointly optimized in an end-to-end manner.

---

## Key Features

- **Structure–attribute joint learning** for integrating gene-expression and cell-topology information.
- **Multi-scale topology modeling** for capturing complementary neighborhood structures.
- **Progressive hierarchical fusion** for multi-level interaction between structural and attribute representations.
- **Cross-view clustering consistency** for coordinating topology-aware and fused predictions.
- **Unsupervised cell clustering** without requiring cell-type annotations during training.
- **Biological validation** through downstream analyses including marker genes, differential expression, enrichment analysis, and PPI analysis.

---

## Data Availability

All datasets used in the manuscript are publicly available from **GEO**, **SRA**, **Figshare**, or the corresponding original repositories.

Detailed dataset information, accession numbers, preprocessing procedures, and references are provided in the manuscript.

---

## Code Availability

The complete source code and analysis scripts will be publicly released after formal publication of the manuscript.

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
