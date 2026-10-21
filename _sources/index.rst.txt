.. liberata_metrics documentation master file, created by
   sphinx-quickstart on Mon Dec  1 23:50:02 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2
   :hidden:

   Home <self>
   Getting Started <getting_started>
   API Reference <api/index>

Liberata Scientometrics
=======================

Welcome to the **Liberata Scientometrics** documentation.

Liberata models academic publishing as a marketplace with contribution shares based credit attribution and provides computational tools to analyze how knowledge and influence flow through citation networks.

.. raw:: html

    <div class="liberata-callout">
        <strong>What is Liberata?</strong><br/>
        Liberata is an Open Access publishing platform with incentivized quality controls.
        Liberata metrics provide methods for quantifying research impact across all research roles, 
        including peer reviewing and replication by using a shares-based credit attribution system, and a marketplace 
        for these academic services where researchers accrue impact from papers in the form of citations in proportion to
        their contributions.
    </div>


.. _intro:

What is Liberata Scientometrics?
--------------------------------

The Liberata Scientometrics library provides tools for analyzing academic knowledge systems as networks of academic impact flow:

- **Papers** are nodes that produce citations
- **Researchers** accrue capital (influence/impact) from papers
- **Citations** are edges showing how capital flows between papers
- **Metrics** quantify capital concentration, returns, risk, and system health

Use cases
---------

**🔗 Academic research**

- Analyze citation networks and knowledge flow
- Create publication-quality visualizations
- Publish studies on how academic capital concentrates in publishing
- Compare different capital allocation policies
- Predict impact and influence of new papers

**🧪 System design**

- Generate synthetic data for testing
- Connect to production data via Supabase
- Evaluate policies for allocating research funding
- Test incentive mechanisms before deployment
- Benchmark algorithm performance on standard datasets

**📊 Portfolio analysis**

- Compute portfolio metrics on research collections
- Track how capital accumulates over time
- Analyze which research areas generate the most impact
- Identify high-risk/high-reward research directions
- Optimize resource allocation across a portfolio of projects

**📚 Education**

- Understand network science and scientometrics
- Learn Python for data science and network analysis
- Explore graph algorithms and sparse matrix computation


Citation
--------

If you use this work in research, please cite the paper and the software:

**Paper:**

.. code-block:: bibtex

    @misc{zhang2026liberatagraphscientometrics,
          title={Liberata -- Graph Scientometrics for a Share Based System of Academic Publishing}, 
          author={Han Zhang and Anshuman Sabath and Timothy W. Dunn and L. Catherine Brinson},
          year={2026},
          eprint={2605.02128},
          archivePrefix={arXiv},
          primaryClass={cs.DL},
          url={https://arxiv.org/abs/2605.02128}, 
    }

**Software:**

.. code-block:: bibtex

    @software{liberata_scientometrics_2025,
        title={Liberata Scientometrics: A package for analyzing academic capital flow},
        author={Wang, Hanlin and Saha Choudhury, Arjun and Wang, Derek and Sabath, Anshuman and Roongta, Aarsh and Knittel, Clayton},
        year={2025},
        url={https://github.com/Liberata-Academic-Publishing/liberata-metrics}
    }


.. _modules-section:
.. _api-ref:

Additional Resources
--------------------

- **Examples & Tutorials**: Use `test_scripts/` for runnable examples.
- **Synthetic Data Example**: `test_scripts/matrix_generators_test.py`
- **Portfolio Metrics Example**: `test_scripts/portfolio_metrics_test.py`
- **Generator Config Example**: `test_scripts/config/matrix_config.yaml`
- **Related Projects**: Liberata Platform and Liberata Simulations


Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


License
-------

Liberata Scientometrics is released under the Apache License 2.0.
See `LICENSE <https://github.com/Liberata-Academic-Publishing/liberata-metrics/blob/main/LICENSE>`_ for details.


Questions or Feedback?
----------------------

- **Issues**: Report bugs or request features on `GitHub Issues <https://github.com/Liberata-Academic-Publishing/liberata-metrics/issues>`_
- **Discussions**: Ask questions on `GitHub Discussions <https://github.com/Liberata-Academic-Publishing/liberata-metrics/discussions>`_
- **Email**: Contact the development team


Last Updated
------------

Version 0.15.1 (Development)

See `CHANGELOG <https://github.com/Liberata-Academic-Publishing/liberata-metrics/blob/main/CHANGELOG.md>`_ for version history.
