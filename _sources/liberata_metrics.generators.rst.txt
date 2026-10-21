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
    :returns: Tuple of (references, manuscript_ids, manuscript_index_map, upload_dates, manuscript_meta, primary_memberships, all_memberships, topic_index_map)


Submodules
----------

.. toctree::
   :maxdepth: 2

   liberata_metrics.generators.generate_matrices


Common Usage Patterns
---------------------

**Generate data with specific properties:**

.. code-block:: python

    from liberata_metrics.generators import generate_references_matrix

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


Module Contents
---------------

.. automodule:: liberata_metrics.generators
   :members:
   :undoc-members:
   :show-inheritance:
