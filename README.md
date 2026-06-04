# AE-NS: Auto-Encoding Network with Semantic Constraints for Fair Classification

> **Paper:** *AE-NS: An Auto-Encoding Network with Semantic Constraints for Robust Fair Classification on Tabular Data*
## Overview

AE-NS is a multi-task learning framework for algorithmic fairness on tabular data. It jointly trains:

1. A **transformer encoder** producing a disentangled latent representation *Z*
2. A **classification head** minimising cross-entropy
3. A **reconstruction head** preserving feature information in *Z*
4. A **Lagrangian dual variable** that adaptively penalises demographic-parity violation
5. An **orthogonality loss** that decorrelates *Z* from the protected attribute

Training improvements in v2 (label smoothing, gradual warm-up, cosine annealing) and a **Fairlearn post-processing pipeline** on the disentangled latent *Z* achieve Pareto-dominant fairness-accuracy trade-offs.

### Key Results (v2, 3 seeds, ε = 0.01)

| Dataset | Method | Accuracy | DPD ↓ |
|---------|--------|----------|-------|
| Adult | AE-NS v2 | 0.781 ± 0.027 | 0.025 ± 0.016 |
| Adult | AE-NS Z + Fairlearn | 0.824 ± 0.002 | **0.009 ± 0.006** |
| COMPAS | AE-NS v2 | 0.640 ± 0.011 | 0.145 ± 0.004 |
| COMPAS | AE-NS Z + Fairlearn | 0.653 ± 0.012 | **0.035 ± 0.029** |
| Credit | AE-NS v2 | 0.797 ± 0.029 | 0.025 ± 0.009 |
| Credit | AE-NS Z + Fairlearn | 0.820 ± 0.003 | **0.009 ± 0.006** |

## Repository Structure

```
AENS/
├── aens_framework.py          # Core framework (encoder, decoder, classifier, dual optimisation)
├── run_experiments.py         # v1 multi-seed experiment runner (8 baselines, sensitivity, ablation)
├── run_v2_all.py              # v2 experiment runner (Fairpost, NMI, sensitivity, ablation, t-SNE)
├── run_v2_remaining.py        # Completed NMI, classify-on-Z, t-SNE, credit ablation
├── run_v2_fix_compas.py       # Re-ran COMPAS with correct pos_weight
├── run_v2_fix_adult_credit.py # Re-ran Adult/Credit with correct pos_weight
├── make_figures.py            # v1 figure generation
├── make_figures_v2.py         # v2 figure generation (publication figures)
├── verify_numbers.py          # Cross-check paper numbers against JSON files
├── show_results.py            # Print compiled v2 results
├── AE_NS_Experiments_Colab.ipynb  # Colab notebook
├── AE_NS_Results_v1/          # v1 experiment results
│   ├── adult/compas/credit/   # Per-dataset JSONs (baselines, sensitivity, ablation)
│   └── *.csv                  # Summary tables
├── AE_NS_Results_v2/          # v2 experiment results (correct pos_weight)
│   ├── v2_{ds}_results.json   # AE-NS v2 standalone (3 seeds)
│   ├── fairpost_3seeds_{ds}.json  # AE-NS Z + Fairlearn (3 seeds)
│   ├── {ds}_revision.json     # NMI + classify-on-Z
│   ├── {ds}_sensitivity_{alpha,beta}.json
│   ├── {ds}_ablation.json
│   ├── tsne_data_{ds}.npz     # t-SNE embeddings
│   └── pareto_sweep_{ds}.json # Lagrangian config sweep
└── figures/                   # 15 PDF + 15 PNG publication figures
    ├── fig1_pareto.{pdf,png}
    ├── fig2_sensitivity_{alpha,beta}_{ds}.{pdf,png}
    ├── fig2_heatmap_alpha_{ds}.{pdf,png}
    ├── fig3_ablation.{pdf,png}
    ├── fig4_tsne_{ds}.{pdf,png}
    └── fig5_nmi_classify.{pdf,png}
```

## Setup

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 (CUDA recommended)
- scikit-learn, fairlearn, xgboost, matplotlib, seaborn, pandas, numpy

### Installation

```bash
pip install torch scikit-learn fairlearn xgboost matplotlib seaborn pandas numpy openpyxl xlrd
```

## Usage

### Running Experiments

```bash
# v1 baselines + sensitivity + ablation (8 baselines × 3 datasets × 3 seeds)
python run_experiments.py

# v2 with correct pos_weight (Fairpost, NMI, sensitivity, ablation, t-SNE)
python run_v2_all.py

# Fix COMPAS / Adult-Credit with correct pos_weight
python run_v2_fix_compas.py
python run_v2_fix_adult_credit.py
```

### Generating Figures

```bash
python make_figures_v2.py   # Saves 30 files to figures/
```

### Verifying Paper Numbers

```bash
python verify_numbers.py    # Cross-checks all 12 key numbers in paper vs JSONs
```

## Datasets

| Dataset | Samples | Features | Protected | Task |
|---------|---------|----------|-----------|------|
| Adult Income | 48,842 | 14 | Gender | Income > $50K |
| COMPAS | 7,214 | 7 | Race | Recidivism |
| Credit Default | 30,000 | 23 | Age | Default payment |

## Hyperparameters (v2)

| Parameter | Value |
|-----------|-------|
| Optimiser (model) | Adam, lr=1e-3, weight_decay=1e-5 |
| Optimiser (dual v) | Adam, lr=1e-1 |
| Batch size | 64 |
| Max epochs | 150 |
| Early stopping patience | 15 |
| LR scheduler | CosineAnnealingLR (1e-3 → 1e-5) |
| Label smoothing | 0.03 |
| Warm-up epochs | 5 |
| α (prediction weight) | 1.0 |
| β (reconstruction weight) | 0.05 |
| Fairness tolerance ε | 0.01 |
| Seeds | {42, 123, 456} |
| Hardware | NVIDIA RTX 4070 (8.59 GB), CUDA 12.1 |

## Citation

```bibtex
@article{hasan2026aens,
  title={AE-NS: An Auto-Encoding Network with Semantic Constraints for Robust Fair Classification on Tabular Data},
  author={Hasan, Raza and others},
  journal={AODS},
  year={2026}
}
```

## License

MIT
