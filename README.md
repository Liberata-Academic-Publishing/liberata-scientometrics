# liberata-scientometrics

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

If you use this package in research, please cite:

```bibtex
@software{liberata_scientometrics_2025,
    title={Liberata Scientometrics: A package for analyzing academic capital flow},
    author={Wang, Derek and Huo, Chuying and Sabath, Anshuman and Wang, Hanlin},
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

## Legacy Setup Notes (Preserved)

The following original notes are preserved to follow open-close editing and maintain backward continuity for existing contributors.

<details>
<summary>Original README notes (preserved verbatim)</summary>

A comprehensive Python package for computing different metrics in the Liberata System.

### Version 0.15.1

In development module for Liberata metrics compuation functions, currently written under the assumption that the capital matrix in COO, plus the ID-index mapping for contributors and manuscripts, are available as input.

### Install & Use
We highly recommend using a virtual env, either `conda` or `uv`, or any of your choice. Example showing conda environment creation

```bash
conda create -n liberata python=3.11
```

#### Clone Locally
Use `git clone` to clone the repository locally at your desired location.
```bash
conda activate liberata
cd liberata-scientometrics
pip install -r requirements.txt
pip install -e .
python test_scripts/matrix_generators_test.py #(this will create the appropriate matrices needed for tests)
python test_scripts/portfolio_metrics_test.py #(change the BASE_DIR to point to the right location, and run tests for computing metrics)
```

#### Direct install
You can also directly install the module using `git+https`

```bash
conda activate liberata
pip install git+https://github.com/Liberata-Academic-Publishing/liberata-scientometrics
python test_scripts/matrix_generators_test.py #(this will create the appropriate matrices needed for tests)
python test_scripts/portfolio_metrics_test.py #(change the BASE_DIR to point to the right location, and run tests for computing metrics)
```

## Local Testing
For local testing, navigate to the ``test_scripts/`` directory and run

### Toy Data Generation
Liberata classifies manuscript topics based on [OpenAlex](https://help.openalex.org/hc/en-us/articles/24736129405719-Topics) topics. You can download the list of topics to generate synthetic graphs based on a sample of OpenAlex topics (check `data/download_data.txt`), or, just use generic topic names (default). To run local tests, change or add your configuration in `test-scripts/config/matrix_config.yaml`  
```
python test_scripts/matrix_generators_test.py
```

The outputted matrices are stored in COO in the ``test_scripts/output/`` directory. 

### Portfolio Metrics Function tests
Update the path names in ``test_scripts/portfolio_metrics_test.py`` to point to the appropriate directory storing the toy matrices. Then from project root, run
```
python test_scripts/portfolio_metrics_test.py
```

### Build Documentation locally
```
sphinx-apidoc -o docs/source/api/generated src/liberata_metrics -f --separate

sphinx-build -b html docs/source docs/build/html

sphinx-autobuild docs/source docs/build/html # Open http://127.0.0.1:8000 in browser
```

</details>
