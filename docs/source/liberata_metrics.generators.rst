liberata\_metrics.generators package
====================================

Overview
--------

The generators module provides tools for creating synthetic academic knowledge graphs and capital matrices. This is essential for:

- **Testing**: Validate metrics computation without relying on real data
- **Benchmarking**: Compare algorithm performance on controlled datasets
- **Experimentation**: Explore "what-if" scenarios with specific properties
- **Education**: Understand how the Liberata system works with transparent synthetic data


Data Structures Generated
--------------------------

The generators create several interconnected data structures:

**References Matrix**
    A sparse matrix encoding citation relationships between manuscripts.
    
    - Shape: (num_manuscripts, num_manuscripts)
    - Entries: Number of times manuscript i cites manuscript j
    - Format: COO (coordinate format) for memory efficiency
    - Sparsity: Controlled via `citation_density` parameter

**Capital Matrix**
    Represents how manuscripts generate academic capital that contributors accrue.
    
    - Shape: (num_manuscripts, num_contributors)
    - Entries: Capital accrued by contributor j from manuscript i
    - Format: COO sparse matrix
    - Extended format: May include multiple capital blocks for different capital types

**ID Mappings**
    Dictionaries mapping human-readable IDs to matrix indices:
    
    - `manuscript_id_to_index`: Maps manuscript UUIDs to row indices
    - `contributor_id_to_index`: Maps contributor UUIDs to column indices
    - Used for linking computed metrics back to real entities

**Metadata**
    Supplementary information about generated entities:
    
    - Manuscript metadata: upload dates, topics, retraction status
    - Contributor metadata: attributes and relationships
    - Topic assignments based on OpenAlex topics


Key Functions
-------------

.. py:function:: generate_references_matrix(num_manuscripts, citation_density=0.05, start_date=date(2020, 1, 1), end_date=date(2024, 1, 1), seed=None)

    Generate a synthetic citation network.
    
    :param num_manuscripts: Number of manuscripts to generate
    :param citation_density: Sparsity of citation relationships (0-1)
    :param start_date: Earliest manuscript publication date
    :param end_date: Latest manuscript publication date
    :param seed: Random seed for reproducibility
    :returns: Tuple of (references_matrix, manuscript_ids, id_to_index_map, dates_map, metadata_df, capital_matrix, contributor_matrix)
    
    Example:
    
    .. code-block:: python
    
        references, ms_ids, ms_map, dates, meta, capital, contribs = \
            generate_references_matrix(num_manuscripts=500, citation_density=0.02)


Configuration
-------------

Generators are configured via YAML files. Example `matrix_config.yaml`:

.. code-block:: yaml

    # Number of manuscripts to generate
    num_manuscripts: 1000
    
    # Citation network sparsity (0 = no citations, 1 = complete graph)
    citation_density: 0.05
    
    # Time range for manuscript generation
    start_date: 2020-01-01
    end_date: 2024-12-31
    
    # Random seed for reproducibility
    seed: 42
    
    # Use OpenAlex topics (True) or generic names (False)
    use_openalex_topics: True
    
    # Output format and location
    output_dir: ./test_scripts/output
    output_format: coo  # Sparse coordinate format


Submodules
----------

.. toctree::
   :maxdepth: 2

   liberata_metrics.generators.generate_matrices


Common Usage Patterns
---------------------

**Generate test data for unit tests:**

.. code-block:: python

    from liberata_metrics.generators import generate_references_matrix
    
    # Create reproducible test data
    refs, ms_ids, ms_map, dates, meta, cap, contribs = \
        generate_references_matrix(
            num_manuscripts=100,
            citation_density=0.03,
            seed=12345  # Fixed seed for reproducibility
        )
    
    # Use in tests
    assert refs.shape[0] == 100
    assert len(ms_ids) == 100

**Generate data with specific properties:**

.. code-block:: python

    # Sparse citation network
    sparse_refs, *_ = generate_references_matrix(
        num_manuscripts=1000,
        citation_density=0.001  # Very sparse
    )
    
    # Dense network for comparing algorithms
    dense_refs, *_ = generate_references_matrix(
        num_manuscripts=1000,
        citation_density=0.1  # Much denser
    )

**Load from configuration file:**

.. code-block:: python

    import yaml
    from liberata_metrics.generators import generate_references_matrix
    
    with open('config/matrix_config.yaml') as f:
        config = yaml.safe_load(f)
    
    refs, *_ = generate_references_matrix(**config)


Module Contents
---------------

.. automodule:: liberata_metrics.generators
   :members:
   :undoc-members:
   :show-inheritance:
