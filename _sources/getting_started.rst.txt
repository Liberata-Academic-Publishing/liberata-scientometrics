Getting started
===============

.. _installation:

Installation
------------

**From GitHub**

.. code-block:: bash

    pip install git+https://github.com/Liberata-Academic-Publishing/liberata-metrics.git

**Local development**

.. code-block:: bash

    git clone https://github.com/Liberata-Academic-Publishing/liberata-metrics.git
    cd liberata-metrics
    pip install -r requirements.txt
    pip install -e .


Data requirements
-----------------

To use this package, you need:

1. **References Matrix**: Citation relationships between papers

   - Sparse matrix format (COO, CSR, or CSC)
   - Shape: (num_papers, num_papers)
   - Entry [i,j] = number of times paper i cites paper j

2. **Capital Matrix**: Capital accrued by researchers from papers

   - Sparse matrix format
   - Shape: (num_papers, num_contributors)
   - Entry [i,j] = capital accrued by contributor j from paper i
   - Internally depends on the Shares Matrix

3. **ID Mappings**: Link matrix indices back to identifiers

   - Paper IDs to row indices
   - Contributor IDs to column indices
   - Timestamps for temporal analysis

The ``generators`` module can create synthetic data. For production data, use the ``integrations.supabase`` module to connect to Liberata's database.


.. _example:

Example
-------

Generate a small citation network, split credit across contributors, and total the academic capital.

.. code-block:: python

    from liberata_metrics.generators import (
        generate_references_matrix,
        generate_shares_matrix,
        build_capital_matrix,
    )
    from liberata_metrics.metrics.portfolio_metrics import academic_capital

    # Citation network. One row and one column per manuscript.
    references, manuscript_ids, manuscript_index_map, upload_dates, manuscript_meta, \
        primary_memberships, all_memberships, topic_index_map = generate_references_matrix(
            num_manuscripts=50,
            citation_density=0.08,
            seed=42,
        )

    # How each manuscript's credit is split across contributors.
    shares, contributor_ids, contributor_index_map = generate_shares_matrix(
        manuscript_ids=manuscript_ids,
        manuscript_index_map=manuscript_index_map,
        num_contributors=10,
        avg_contributors_per_man=3,
        std_contributors_per_man=1,
        seed=42,
    )

    # Capital is each contributor's share, weighted by citations.
    capital = build_capital_matrix(references, shares)

    # Sum of academic capital for every contributor in the map.
    total = academic_capital(capital, contributor_index_map)
    print(f"{len(manuscript_ids)} manuscripts, {len(contributor_ids)} contributors")
    print(f"Total academic capital: {total:.2f}")
