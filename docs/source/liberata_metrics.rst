liberata\_metrics package
=========================

The Liberata Scientometrics Library
-----------------------------------

**liberata_metrics** is a comprehensive Python package for computing metrics on academic knowledge graphs and analyzing the flow of academic capital in publishing systems.

**What is it?**

Liberata models academic knowledge production as a capital allocation system where:

- Researchers (contributors) accrue academic capital from their papers (manuscripts)
- Papers cite other papers, creating a web of influence
- Capital flows through citations, accumulating value over time
- Metrics quantify how capital and influence distribute across the system

This package provides the computational tools to analyze these dynamics.

**Key Features**

- **Portfolio Metrics**: Analyze manuscript collections like financial portfolios (returns, risk, correlations, Sharpe ratios)
- **Market Dynamics**: Study how capital flows and concentrates across the system
- **Network Analysis**: Compute graph-based metrics on citation networks
- **Synthetic Data**: Generate realistic test data for validation and experimentation
- **Production Integration**: Connect to Supabase for real-world data
- **Publication-Ready Visualizations**: Create figures for papers and presentations


Main Components
---------------

.. toctree::
   :maxdepth: 2
   :caption: Core Modules
   :hidden:

   liberata_metrics.metrics
   liberata_metrics.generators
   liberata_metrics.utils
   liberata_metrics.visualizations
   api/liberata_metrics.integrations


Quick Start Example
-------------------

Here's a minimal example computing portfolio metrics:

.. code-block:: python

    from liberata_metrics.generators import generate_references_matrix
    from liberata_metrics.metrics.portfolio_metrics import PortfolioMetrics
    import matplotlib.pyplot as plt
    
    # Step 1: Generate synthetic data
    print("Generating synthetic citation network...")
    refs, ms_ids, ms_map, dates, meta, capital, contribs = \
        generate_references_matrix(
            num_manuscripts=500,
            citation_density=0.03,
            seed=42
        )
    print(f"Generated {len(ms_ids)} manuscripts with {capital.shape[1]} contributors")
    
    # Step 2: Compute metrics
    print("Computing portfolio metrics...")
    pm = PortfolioMetrics(capital)
    
    total_cap = pm.total_capital()
    volatility = pm.compute_volatility()
    correlation = pm.compute_correlation_matrix()
    
    print(f"Total academic capital: {total_cap:.2f}")
    print(f"Portfolio volatility: {volatility:.4f}")
    print(f"Correlation matrix shape: {correlation.shape}")
    
    # Step 3: Analyze
    print("Computing returns...")
    returns = pm.compute_returns()
    sharpe = pm.compute_sharpe_ratio(returns)
    print(f"Sharpe ratio: {sharpe:.4f}")
    
    # Step 4: Visualize
    from liberata_metrics.visualizations import matrix_visuals
    
    fig, ax = plt.subplots()
    matrix_visuals.plot_matrix_heatmap(
        capital[:50, :50],  # Subset for clarity
        title='Capital Allocation (First 50 manuscripts)',
        ax=ax
    )
    plt.tight_layout()
    plt.show()

**Output:**

::

    Generating synthetic citation network...
    Generated 500 manuscripts with 150 contributors
    Computing portfolio metrics...
    Total academic capital: 12534.50
    Portfolio volatility: 0.0342
    Correlation matrix shape: (150, 150)
    Computing returns...
    Sharpe ratio: 2.1456


Use Cases
---------

**Academic Research**
    - Publish studies on how academic capital concentrates in publishing
    - Compare different capital allocation policies
    - Predict impact and influence of new papers

**System Design**
    - Evaluate policies for allocating research funding
    - Test incentive mechanisms before deployment
    - Benchmark algorithm performance on standard datasets

**Portfolio Analysis**
    - Analyze which research areas generate the most impact
    - Identify high-risk/high-reward research directions
    - Optimize resource allocation across a portfolio of projects

**Educational**
    - Understand network science and scientometrics
    - Learn Python for data science and network analysis
    - Explore graph algorithms and sparse matrix computation


Data Requirements
------------------

To use this package, you need:

1. **References Matrix**: Citation relationships between papers
   
   - Sparse matrix format (COO, CSR, or CSC)
   - Shape: (num_papers, num_papers)
   - Entry [i,j] = number of times paper i cites paper j

2. **Capital Matrix**: Capital accrued by researchers from papers
   
   - Sparse matrix format
   - Shape: (num_papers, num_contributors)
   - Entry [i,j] = capital accrued by contributor j from paper i
   - Internally depends on the Shares Matrix

3. **ID Mappings**: Link matrix indices back to identifiers
   
   - Paper IDs to row indices
   - Contributor IDs to column indices
   - Timestamps for temporal analysis

The `generators` module can create synthetic data. For production data, use the `integrations.supabase` module to connect to Liberata's database.


Performance Characteristics
---------------------------

- **Scalability**: Handles millions of papers and thousands of contributors
- **Memory**: Sparse matrix format keeps memory usage proportional to non-zero entries
- **Speed**: Vectorized NumPy operations for efficient computation
- **Reproducibility**: Deterministic results with seed control


Getting Help
------------

- **API Documentation**: Browse the complete reference above
- **Examples**: Check `test_scripts/` for working examples
- **Testing**: Run `python test_scripts/portfolio_metrics_test.py` to validate setup
- **Issues**: Report bugs on GitHub


Installation
------------

Install from PyPI:

.. code-block:: bash

    pip install liberata-scientometrics

Or from GitHub with the latest development version:

.. code-block:: bash

    pip install git+https://github.com/Liberata-Academic-Publishing/liberata-scientometrics

For development, clone the repository and install in editable mode:

.. code-block:: bash

    git clone https://github.com/Liberata-Academic-Publishing/liberata-scientometrics
    cd liberata-scientometrics
    pip install -e .


Citation
--------

If you use this package in research, please cite:

::

    @software{liberata_scientometrics,
        title={Liberata Scientometrics: A package for analyzing academic capital flow},
        author={Wang, Derek and Huo, Chuying and Sabath, Anshuman and Wang, Hanlin},
        year={2025},
        url={https://github.com/Liberata-Academic-Publishing/liberata-scientometrics}
    }


Module Contents
---------------

.. automodule:: liberata_metrics
   :members:
   :undoc-members:
   :show-inheritance:
