API Reference
=============

The Liberata Scientometrics library provides a comprehensive set of modules for analyzing and computing metrics on academic knowledge graphs and portfolios. This section documents the complete API.

Core Modules
------------

The library is organized into four main components:

**Metrics** (:doc:`generated/liberata_metrics.metrics`)
    Compute various scientometric metrics on your data:
    
    - **Portfolio Metrics**: Analyze manuscript portfolios, including returns, risk, correlations, and spectral properties
    - **Market Metrics**: Evaluate market dynamics and portfolio performance in the academic capital system
    - **Distribution Metrics**: Analyze distributions of contributors and capital allocation
    - **Graph Metrics**: Compute network-based metrics on citation and collaboration graphs
    - **System Health Metrics**: Monitor the overall health of the academic capital system

**Generators** (:doc:`generated/liberata_metrics.generators`)
    Create synthetic data for testing and analysis:
    
    - Generate reference matrices from scratch
    - Build contributor-manuscript relationships
    - Create topic assignments based on OpenAlex topics
    - Produce COO-format sparse matrices for efficient computation

**Utilities** (:doc:`generated/liberata_metrics.utils`)
    Helper functions for common tasks:
    
    - Data loading from Supabase and local sources
    - Data wrangling and transformation
    - Matrix operations and utilities
    - Sparse matrix utilities

**Visualizations** (:doc:`generated/liberata_metrics.visualizations`)
    Create visual representations of your data:
    
    - Matrix visualizations (heatmaps, sparsity plots)
    - Time series visualizations
    - Network and graph visualizations
    - Portfolio analysis plots

**Integrations** (:doc:`generated/liberata_metrics.integrations`)
    Connect to external systems:
    
    - Supabase integration for production data access
    - Logging and configuration


Quick Reference
---------------

Common workflows:

1. **Generate test data**:
   
   .. code-block:: python
   
       from liberata_metrics.generators import generate_references_matrix
       
       matrices = generate_references_matrix(
           num_manuscripts=100,
           citation_density=0.05
       )

2. **Compute portfolio metrics**:
   
   .. code-block:: python
   
       from liberata_metrics.metrics import PortfolioMetrics
       
       pm = PortfolioMetrics(capital_matrix)
       returns = pm.compute_returns()
       risk = pm.compute_risk()

3. **Analyze system dynamics**:
   
   .. code-block:: python
   
       from liberata_metrics.metrics import MarketMetrics
       
       mm = MarketMetrics(references, capital)
       efficiency = mm.compute_efficiency()


Full API Documentation
-----------------------

.. toctree::
   :maxdepth: 2
   :caption: Complete Reference

   modules
   generated/modules
