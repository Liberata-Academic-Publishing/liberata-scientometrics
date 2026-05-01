liberata\_metrics.metrics package
=================================

Overview
--------

The metrics module is the heart of the Liberata Scientometrics library. It provides a comprehensive suite of functions and classes for computing scientometric metrics on academic capital systems.

**What are these metrics?**

The Liberata system models academic knowledge production as a capital accrual system, where:

- **Manuscripts** are publications that produce citations
- **Contributors** are researchers who accrue academic capital from manuscripts
- **Capital** represents the influence and returns that researchers accrue from their work
- **References** encode the citation relationships between manuscripts

The metrics in this module help you understand:

- How capital flows through the academic system
- Portfolio performance and risk characteristics
- Market dynamics and efficiency
- System health and concentration
- Individual and collective contributions


Key Submodules
---------------

.. toctree::
   :maxdepth: 2

   liberata_metrics.metrics.portfolio_metrics
   liberata_metrics.metrics.market_metrics
   liberata_metrics.metrics.distribution_metrics
   liberata_metrics.metrics.graph
   liberata_metrics.metrics.system_health_metrics
   liberata_metrics.metrics.legacy_metric


Common Patterns
---------------

Most metric computation follows this pattern:

1. **Prepare data**: Load or generate references and capital matrices
2. **Create metric object**: Instantiate a metric class with your data
3. **Compute metrics**: Call computation methods to get results
4. **Analyze results**: Interpret and visualize the output

Example:

.. code-block:: python

    from liberata_metrics.metrics.portfolio_metrics import PortfolioMetrics
    import numpy as np
    from scipy import sparse
    
    # Create example capital matrix (manuscripts × contributors)
    capital = sparse.random(100, 50, density=0.1)
    
    # Compute metrics
    pm = PortfolioMetrics(capital)
    total_capital = pm.total_capital()
    volatility = pm.compute_volatility()


Module Contents
---------------

.. automodule:: liberata_metrics.metrics
   :members:
   :undoc-members:
   :show-inheritance:
