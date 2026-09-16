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


Common Tasks
------------

**Loading Data from Supabase**

.. code-block:: python

    from liberata_metrics.utils import load_supabase_data
    
    # Load references and capital matrices from production
    references, capital = load_supabase_data.fetch_matrices()
    
    # Load specific time period
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    refs_period, cap_period = load_supabase_data.fetch_matrices_for_period(
        start_date, end_date
    )

**Loading from Local Files**

.. code-block:: python

    from liberata_metrics.utils import data_loading
    from scipy import sparse
    import numpy as np
    
    # Load COO matrices
    references = data_loading.load_coo_matrix('data/references.npz')
    capital = data_loading.load_coo_matrix('data/capital.npz')
    
    # Load metadata
    id_maps = data_loading.load_id_mappings('data/mappings.json')

**Data Wrangling**

.. code-block:: python

    from liberata_metrics.utils import data_wrangling
    
    # Filter matrix for a subset of manuscripts
    subset_references = data_wrangling.filter_manuscripts(
        references, 
        manuscript_ids
    )
    
    # Normalize capital allocation
    normalized_capital = data_wrangling.normalize_capital(capital)
    
    # Aggregate by topic
    topic_aggregated = data_wrangling.aggregate_by_topic(
        capital,
        topic_mappings
    )

**Sparse Matrix Operations**

.. code-block:: python

    from liberata_metrics.utils import utils
    import numpy as np
    
    # Safe division on sparse matrices
    result = utils.sparse_divide(numerator, denominator)
    
    # Extract submatrix
    sub = utils.submatrix(matrix, row_indices, col_indices)
    
    # Convert between sparse formats
    lil_matrix = utils.to_lil(coo_matrix)
    csr_matrix = utils.to_csr(coo_matrix)


Performance Considerations
---------------------------

**Working with Large Matrices**

The utils module is optimized for large sparse matrices:

- Sparse matrix format (COO, CSR, CSC) to minimize memory usage
- Lazy operations that avoid creating dense copies
- Vectorized NumPy operations for speed

.. code-block:: python

    from liberata_metrics.utils import data_loading
    import scipy.sparse as sparse
    
    # Load large matrix efficiently
    large_matrix = data_loading.load_coo_matrix('data/large.npz')
    
    # Convert to efficient format for operations
    csr_matrix = large_matrix.tocsr()  # For row operations
    csc_matrix = large_matrix.tocsc()  # For column operations
    
    # Avoid creating dense copies
    result = csr_matrix.multiply(another_sparse)  # Stays sparse

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
