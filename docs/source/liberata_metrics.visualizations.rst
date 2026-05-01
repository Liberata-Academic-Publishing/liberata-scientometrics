liberata\_metrics.visualizations package
======================================

Overview
--------

The visualizations module provides tools for creating publication-quality visualizations of academic knowledge graphs and portfolio metrics. Visualizations help communicate findings and identify patterns in complex data.

**Why Visualizations Matter**

Visual analysis helps you:

- **Explore data structure**: Understand the sparsity and scale of networks
- **Identify patterns**: Spot clusters, hierarchies, and anomalies
- **Communicate results**: Create compelling figures for presentations and papers
- **Validate computations**: Visually verify that metrics behave as expected
- **Compare scenarios**: Side-by-side analysis of different conditions


Submodules
----------

.. toctree::
   :maxdepth: 2

   liberata_metrics.visualizations.matrix_visuals
   liberata_metrics.visualizations.time_series_visuals


Visualization Types
-------------------

**Matrix Visualizations**

Understand the structure of your sparse matrices:

.. code-block:: python

    from liberata_metrics.visualizations import matrix_visuals
    import matplotlib.pyplot as plt
    
    # Heatmap of capital allocation
    fig, ax = plt.subplots(figsize=(12, 8))
    matrix_visuals.plot_matrix_heatmap(
        capital_matrix,
        title='Capital Allocation Heatmap',
        ax=ax
    )
    plt.show()
    
    # Sparsity pattern
    fig, ax = plt.subplots()
    matrix_visuals.plot_sparsity_pattern(
        references_matrix,
        title='Citation Network Sparsity',
        ax=ax
    )
    plt.show()
    
    # Degree distribution
    fig, ax = plt.subplots()
    matrix_visuals.plot_degree_distribution(
        references_matrix,
        ax=ax
    )
    plt.show()

**Time Series Visualizations**

Track how metrics evolve over time:

.. code-block:: python

    from liberata_metrics.visualizations import time_series_visuals
    import matplotlib.pyplot as plt
    
    # Portfolio returns over time
    fig, ax = plt.subplots()
    time_series_visuals.plot_returns_timeseries(
        returns_df,
        title='Portfolio Returns',
        ax=ax
    )
    plt.show()
    
    # Cumulative capital growth
    fig, ax = plt.subplots()
    time_series_visuals.plot_cumulative_capital(
        capital_by_date,
        contributors=None,  # All contributors
        ax=ax
    )
    plt.show()
    
    # Risk metrics over time
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    time_series_visuals.plot_risk_panel(
        volatility_df,
        correlation_df,
        beta_df,
        spectral_df,
        axes=axes
    )
    plt.tight_layout()
    plt.show()


Common Workflows
----------------

**Creating a Publication Figure**

.. code-block:: python

    from liberata_metrics.visualizations import matrix_visuals
    import matplotlib.pyplot as plt
    
    # Create figure with publication-ready settings
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Citation network
    matrix_visuals.plot_sparsity_pattern(
        references,
        title='(a) Citation Network Structure',
        ax=axes[0],
        cmap='Blues'
    )
    
    # Plot 2: Capital allocation
    matrix_visuals.plot_matrix_heatmap(
        capital,
        title='(b) Capital Allocation',
        ax=axes[1],
        vmin=0,
        vmax=capital.max()
    )
    
    # Plot 3: Contributor degree
    matrix_visuals.plot_contributor_activity(
        capital,
        top_n=10,
        title='(c) Top 10 Contributors',
        ax=axes[2]
    )
    
    # Save with high DPI for publication
    plt.tight_layout()
    fig.savefig('figure1.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('figure1.png', dpi=150, bbox_inches='tight')
    plt.show()

**Interactive Exploration**

.. code-block:: python

    from liberata_metrics.visualizations import matrix_visuals
    import plotly.express as px
    
    # Create interactive heatmap (if Plotly support available)
    # Hover over cells to see exact values
    fig = matrix_visuals.plot_matrix_interactive(
        capital,
        title='Interactive Capital Allocation',
        labels={'x': 'Contributor', 'y': 'Manuscript'}
    )
    fig.show()


Best Practices
--------------

**Color Maps**
    Use perceptually uniform colormaps:
    
    - `'viridis'`: Sequential data (default, works for colorblind)
    - `'plasma'`: High contrast
    - `'RdYlBu'`: Diverging (positive/negative values)
    - Avoid `'jet'` (misleads the eye)

**Figure Size and DPI**
    
    - Screen viewing: 72-96 DPI, 6-8" wide
    - Print quality: 300 DPI minimum
    - Paper figures: Typically 3-4" wide (fits in one column)

**Annotation**
    
    Always include:
    
    - Clear title
    - Axis labels with units
    - Color bar with scale
    - Figure caption describing the data and interpretation

**Reproducibility**
    
    .. code-block:: python
    
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Set random seed for reproducible layouts
        import numpy as np
        np.random.seed(42)
        
        # Save figure configuration
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['font.size'] = 10
        plt.rcParams['lines.linewidth'] = 1.5


Module Contents
---------------

.. automodule:: liberata_metrics.visualizations
   :members:
   :undoc-members:
   :show-inheritance:
