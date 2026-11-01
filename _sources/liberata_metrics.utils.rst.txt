liberata\_metrics.utils package
============================

Overview
--------

The utils module provides essential utilities for working with the Liberata Scientometrics library. It includes:

- **Data Loading**: Fetch data from Supabase or local files
- **Data Wrangling**: Transform and prepare data for analysis
- **Matrix Operations**: Efficient utilities for sparse matrix manipulation
- **Helper Functions**: Common operations used across the package


Submodules
----------

.. toctree::
   :maxdepth: 2

   liberata_metrics.utils.load_supabase_data
   liberata_metrics.utils.data_loading
   liberata_metrics.utils.data_wrangling
   liberata_metrics.utils.utils


Performance Considerations
---------------------------

**Working with Large Matrices**

The utils module is optimized for large sparse matrices:

- Sparse matrix format (COO, CSR, CSC) to minimize memory usage
- Lazy operations that avoid creating dense copies
- Vectorized NumPy operations for speed

**Memory Profiling**

.. code-block:: python

    from liberata_metrics.utils import utils
    import sys
    
    # Check size of matrix in memory
    size_mb = sys.getsizeof(matrix) / 1e6
    nnz = matrix.nnz  # Number of non-zero elements
    density = nnz / (matrix.shape[0] * matrix.shape[1])


Module Contents
---------------

.. automodule:: liberata_metrics.utils
   :members:
   :undoc-members:
   :show-inheritance:
