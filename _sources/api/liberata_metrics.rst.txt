liberata\_metrics package
=========================

Overview
--------

This is the top-level package for the Liberata Scientometrics library.
It exposes logging utilities and organizes the core API into domain-specific
subpackages for metrics, generators, utilities, integrations, and visualization.


Package Map
-----------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Package
     - Purpose
     - Reference
   * - ``liberata_metrics.metrics``
     - Portfolio, market, distribution, graph, and system-health metrics.
     - :doc:`Metrics API <liberata_metrics.metrics>`
   * - ``liberata_metrics.generators``
     - Synthetic data generation for references, shares, and capital matrices.
     - :doc:`Generators API <liberata_metrics.generators>`
   * - ``liberata_metrics.utils``
     - Data loading, wrangling, and sparse matrix helper utilities.
     - :doc:`Utils API <liberata_metrics.utils>`
   * - ``liberata_metrics.visualizations``
     - Matrix and time-series visualization helpers.
     - :doc:`Visualizations API <liberata_metrics.visualizations>`
   * - ``liberata_metrics.integrations``
     - External system integrations (including Supabase paths).
     - :doc:`Integrations API <liberata_metrics.integrations>`


Top-Level Public API
--------------------

The top-level ``liberata_metrics`` namespace exposes logging setup helpers.

.. autosummary::
   :toctree: generated

   liberata_metrics.configure_logging
   liberata_metrics.get_logger


Subpackage Reference
--------------------

Use the pages below for full module-level and function-level documentation.

.. toctree::
   :maxdepth: 2

   liberata_metrics.generators
   liberata_metrics.integrations
   liberata_metrics.metrics
   liberata_metrics.utils
   liberata_metrics.visualizations


Internal Modules
----------------

.. toctree::
   :maxdepth: 2

   liberata_metrics.logging


Module Contents
---------------

.. automodule:: liberata_metrics
   :members:
   :undoc-members:
   :show-inheritance:
