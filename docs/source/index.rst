.. liberata_metrics documentation master file, created by
   sphinx-quickstart on Mon Dec  1 23:50:02 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Main

   self
   liberata_metrics
   api/index

Liberata Scientometrics: Academic Knowledge Graph Analysis
=========================================================

Welcome to the **Liberata Scientometrics** library documentation.

Liberata models academic publishing as a capital allocation system and provides computational tools to analyze how knowledge and influence flow through citation networks.

.. raw:: html

    <div style="background: #f0f7ff; padding: 15px; margin: 20px 0; border-radius: 5px; border-left: 4px solid #0066cc;">
        <strong>What is Liberata?</strong><br/>
        A framework for analyzing research impact as a capital system where researchers accrue influence from papers, 
        and citations determine how that capital flows through the network.
    </div>


Getting Started
===============

**New to Liberata?** Start here:

1. :ref:`What is Liberata Scientometrics? <intro>` — Understand the core concepts
2. :ref:`Installation <installation>` — Get the package installed
3. :ref:`Quick Start <quickstart>` — Run your first analysis
4. :ref:`Key Concepts <concepts>` — Learn the terminology
5. :ref:`API Reference <api-ref>` — Detailed function documentation


.. _intro:

What is Liberata Scientometrics?
--------------------------------

The Liberata Scientometrics library provides tools for analyzing academic knowledge systems as networks of capital flow:

- **Papers** are nodes that produce citations
- **Researchers** accrue capital (influence) from papers
- **Citations** are edges showing how capital flows between papers
- **Metrics** quantify capital concentration, returns, risk, and system health

This package helps you:

- 📊 Compute portfolio metrics on research collections
- 🔗 Analyze citation networks and knowledge flow
- 🧪 Generate synthetic data for testing
- 📈 Track how capital accumulates over time
- 💾 Connect to production data via Supabase
- 📉 Create publication-quality visualizations


.. _installation:

Installation
============

**Option 1: From PyPI (recommended)**

.. code-block:: bash

    pip install liberata-scientometrics

**Option 2: From GitHub (latest development version)**

.. code-block:: bash

    pip install git+https://github.com/Liberata-Academic-Publishing/liberata-scientometrics

**Option 3: Local development**

.. code-block:: bash

    git clone https://github.com/Liberata-Academic-Publishing/liberata-scientometrics
    cd liberata-scientometrics
    pip install -e .


.. _quickstart:

Quick Start
===========

Generate synthetic data and compute metrics in 30 seconds:

.. code-block:: python

    from liberata_metrics.generators import generate_references_matrix
    from liberata_metrics.metrics.portfolio_metrics import PortfolioMetrics
    
    # Generate test data
    refs, ms_ids, ms_map, dates, meta, capital, contribs = \
        generate_references_matrix(num_manuscripts=100, seed=42)
    
    # Compute portfolio metrics
    pm = PortfolioMetrics(capital)
    print(f"Total capital: {pm.total_capital():.2f}")
    print(f"Volatility: {pm.compute_volatility():.4f}")
    
    # Visualize
    from liberata_metrics.visualizations import matrix_visuals
    import matplotlib.pyplot as plt
    matrix_visuals.plot_sparsity_pattern(refs)
    plt.show()

**Next steps:**

- Explore :ref:`more examples <examples>`
- Check :ref:`core modules <modules-section>`
- Read about :ref:`key concepts <concepts>`


.. _concepts:

Key Concepts
============

**Capital Matrix**
    A sparse matrix where:
    
    - Rows represent manuscripts (papers)
    - Columns represent contributors (researchers)
    - Entry [i,j] = capital accrued by researcher j from paper i
    - Shape: (num_papers, num_researchers)

**References Matrix**
    Citation relationships:
    
    - Both dimensions are papers
    - Entry [i,j] = number of times paper i cites paper j
    - Encodes the knowledge graph structure
    - Shape: (num_papers, num_papers)

**ID Mappings**
    Dictionaries linking matrix indices to real identifiers:
    
    - Connect computed metrics back to papers/researchers
    - Enable temporal analysis with timestamps
    - Support subsetting and filtering

**Metrics**
    Quantitative measures of system behavior:
    
    - **Portfolio metrics**: returns, risk, correlations (paper-based)
    - **Market metrics**: price discovery, efficiency
    - **Distribution metrics**: concentration, inequality
    - **System metrics**: overall health and dynamics


.. _examples:

Common Usage Patterns
=====================

**Pattern 1: Analyze a portfolio of papers**

.. code-block:: python

    from liberata_metrics.metrics.portfolio_metrics import PortfolioMetrics
    
    # Load or create capital matrix
    pm = PortfolioMetrics(capital_matrix)
    
    # Compute standard metrics
    returns = pm.compute_returns()
    volatility = pm.compute_volatility()
    sharpe = pm.compute_sharpe_ratio(returns)
    
    print(f"Sharpe ratio: {sharpe:.2f}")

**Pattern 2: Load real data from Supabase**

.. code-block:: python

    from liberata_metrics.utils import load_supabase_data
    
    # Fetch production data
    references, capital = load_supabase_data.fetch_matrices()
    
    # Load specific time period
    start = '2023-01-01'
    end = '2024-01-01'
    refs_yr, cap_yr = load_supabase_data.fetch_matrices_for_period(start, end)

**Pattern 3: Generate controlled test data**

.. code-block:: python

    from liberata_metrics.generators import generate_references_matrix
    
    # Sparse network (5 papers cite 2% of other papers on average)
    sparse_refs, *_ = generate_references_matrix(
        num_manuscripts=1000,
        citation_density=0.02,
        seed=42
    )
    
    # Dense network (for comparison)
    dense_refs, *_ = generate_references_matrix(
        num_manuscripts=1000,
        citation_density=0.1,
        seed=42
    )

**Pattern 4: Create visualizations**

.. code-block:: python

    from liberata_metrics.visualizations import matrix_visuals, time_series_visuals
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Citation network structure
    matrix_visuals.plot_sparsity_pattern(
        references,
        title='Citation Network',
        ax=axes[0]
    )
    
    # Capital allocation
    matrix_visuals.plot_matrix_heatmap(
        capital,
        title='Capital Allocation',
        ax=axes[1]
    )
    
    plt.tight_layout()
    plt.show()


.. _modules-section:
.. _modules-section:
.. _api-ref:

Documentation Links
===================

- :doc:`Package Guide <liberata_metrics>`
- :doc:`API Reference <api/index>`

Additional Resources
--------------------

- **Examples & Tutorials**: Use `test_scripts/` for runnable examples.
- **Synthetic Data Example**: `test_scripts/matrix_generators_test.py`
- **Portfolio Metrics Example**: `test_scripts/portfolio_metrics_test.py`
- **Generator Config Example**: `test_scripts/config/matrix_config.yaml`
- **Related Projects**: Liberata Platform and Liberata Simulations


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


License
=======

Liberata Scientometrics is released under the Apache License 2.0.
See `LICENSE <https://github.com/Liberata-Academic-Publishing/liberata-scientometrics/blob/main/LICENSE>`_ for details.


Citation
========

If you use this package in research, please cite:

.. code-block:: bibtex

    @software{liberata_scientometrics_2025,
        title={Liberata Scientometrics: A package for analyzing academic capital flow},
        author={Wang, Derek and Huo, Chuying and Sabath, Anshuman and Wang, Hanlin},
        year={2025},
        url={https://github.com/Liberata-Academic-Publishing/liberata-scientometrics}
    }


Questions or Feedback?
======================

- **Issues**: Report bugs or request features on `GitHub Issues <https://github.com/Liberata-Academic-Publishing/liberata-scientometrics/issues>`_
- **Discussions**: Ask questions on `GitHub Discussions <https://github.com/Liberata-Academic-Publishing/liberata-scientometrics/discussions>`_
- **Email**: Contact the development team


Last Updated
============

Version 0.15.1 (Development)

See `CHANGELOG <https://github.com/Liberata-Academic-Publishing/liberata-scientometrics/blob/main/CHANGELOG.md>`_ for version history.
