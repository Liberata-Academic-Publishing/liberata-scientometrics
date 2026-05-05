# liberata-scientometrics

[![arXiv](https://img.shields.io/badge/arXiv-2605.02128-b31b1b.svg)](https://arxiv.org/abs/2605.02128)

Liberata Scientometrics is a Python package for computing metrics on academic knowledge graphs and analyzing how academic impact flows through citation systems.

Version: `0.15.1` (development)

## Overview

Liberata models academic publishing as a marketplace with contribution shares based credit attribution and provides computational tools to analyze how knowledge and influence flow through citation networks:

- Papers (manuscripts) cite other papers, creating a network of influence.
- Contributors accrue academic capital from authored, reviewed, and replicated work.
- Capital allocation and citation structure can be analyzed with portfolio, market, distribution, graph, and system health metrics.

The library is designed for research workflows that need reproducible matrix-based computation and scalable sparse operations.

## Key Capabilities

- Portfolio-style analysis on manuscript collections (capital totals, concentration, diversity, losses).
- Market and system-dynamics metrics for fairness, risk premiums, and health indicators.
- Graph/network analysis over citation structures.
- Synthetic data generation for controlled experiments and regression testing.
- Utilities for loading and transforming matrices.
- Visualization helpers for matrix and time-series outputs.
- Supabase integration paths for production data workflows.

## Installation

We recommend using a dedicated virtual environment (`conda`, `venv`, or `uv`).

### Option 1: Conda environment setup

```bash
conda create -n liberata python=3.11
conda activate liberata
```

### Option 2: Install from GitHub

```bash
pip install git+https://github.com/Liberata-Academic-Publishing/liberata-scientometrics
```

### Option 3: Local development install

```bash
git clone https://github.com/Liberata-Academic-Publishing/liberata-scientometrics
cd liberata-scientometrics
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from liberata_metrics.generators import (
    generate_references_matrix,
    generate_shares_matrix,
    build_capital_matrix,
)
from liberata_metrics.metrics import academic_capital, portfolio_hhi, portfolio_gini

# 1) Generate synthetic citation structure
(
    references,
    manuscript_ids,
    manuscript_index_map,
    upload_dates,
    manuscript_meta,
    primary_memberships,
    all_memberships,
    topic_index_map,
) = generate_references_matrix(num_manuscripts=200, citation_density=0.03, seed=42)

# 2) Generate contributor shares and build capital matrix
shares, contributor_ids, contributor_index_map = generate_shares_matrix(
    manuscript_ids=manuscript_ids,
    manuscript_index_map=manuscript_index_map,
    num_contributors=300,
    avg_contributors_per_man=5,
    std_contributors_per_man=2,
    seed=42,
)
capital = build_capital_matrix(references, shares)

# 3) Select a contributor subset and compute portfolio metrics
subset = {cid: idx for cid, idx in list(contributor_index_map.items())[:50]}
print("Academic capital:", academic_capital(capital, subset))
print("Portfolio HHI:", portfolio_hhi(capital, subset))
print("Portfolio Gini:", portfolio_gini(capital, subset))
```

## Data Model

Core inputs are sparse matrices plus index mappings:

- `references`: shape `(M, M)` citation matrix between manuscripts.
- `shares`: shape `(M, M + 3C)` manuscript-to-contributor role-weight matrix (authors/reviewers/replicators blocks).
- `capital`: shape `(M, M + 3C)` derived capital allocation matrix.
- `manuscript_index_map`: manuscript ID to row index.
- `contributor_index_map`: contributor ID to contributor-block column index.

`M` = number of manuscripts, `C` = number of contributors.

## Module Guide

- `liberata_metrics.metrics`: portfolio, market, distribution, legacy, graph, and system health metrics.
- `liberata_metrics.generators`: synthetic reference/share generation and time-series capital snapshots.
- `liberata_metrics.utils`: loading, wrangling, sparse helpers, Supabase-oriented loaders.
- `liberata_metrics.visualizations`: matrix and time-series visual utilities.
- `liberata_metrics.integrations`: integration helpers (including Supabase paths).

## Local Testing

### Generate toy data

Liberata classifies manuscript topics using OpenAlex topics. You can use the included topic mapping data or configure your own settings in `test_scripts/config/matrix_config.yaml`.

```bash
python test_scripts/matrix_generators_test.py
```

Generated matrices are written to `test_scripts/output/` in COO format.

### Run metrics tests/scripts

Update `BASE_DIR` in `test_scripts/portfolio_metrics_test.py` to point to a generated output folder, then run:

```bash
python test_scripts/portfolio_metrics_test.py
```

Additional runnable scripts are available in `test_scripts/` for market, distribution, system health, and graph metrics.

## Documentation

Build docs locally:

```bash
sphinx-apidoc -o docs/source/api/generated src/liberata_metrics -f --separate
sphinx-build -b html docs/source docs/build/html
sphinx-autobuild docs/source docs/build/html  # http://127.0.0.1:8000
```

Primary docs entry point:

- `docs/source/index.rst`

## Citation

If you use this work in research, please cite the paper and the software:

**Paper:**

```bibtex
@misc{zhang2026liberatagraphscientometrics,
      title={Liberata -- Graph Scientometrics for a Share Based System of Academic Publishing}, 
      author={Han Zhang and Anshuman Sabath and Timothy W. Dunn and L. Catherine Brinson},
      year={2026},
      eprint={2605.02128},
      archivePrefix={arXiv},
      primaryClass={cs.DL},
      url={https://arxiv.org/abs/2605.02128}, 
}
```

**Software:**

```bibtex
@software{liberata_scientometrics_2025,
    title={Liberata Scientometrics: A package for analyzing academic capital flow},
    author={Wang, Hanlin and Saha Choudhury, Arjun and Wang, Derek and Sabath, Anshuman and Roongta, Aarsh and Knittel, Clayton},
    year={2025},
    url={https://github.com/Liberata-Academic-Publishing/liberata-scientometrics}
}
```

## License

This project is licensed under the Apache License 2.0.
See `LICENSE` for details.

## Support

- GitHub Issues: https://github.com/Liberata-Academic-Publishing/liberata-scientometrics/issues
- GitHub Discussions: https://github.com/Liberata-Academic-Publishing/liberata-scientometrics/discussions