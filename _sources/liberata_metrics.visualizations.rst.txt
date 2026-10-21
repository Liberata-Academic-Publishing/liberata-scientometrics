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
