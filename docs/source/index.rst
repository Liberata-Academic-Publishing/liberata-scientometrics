.. liberata_metrics documentation master file, created by
   sphinx-quickstart on Mon Dec  1 23:50:02 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Liberata Scientometrics Library
==============================

.. Add your content using ``reStructuredText`` syntax. See the
.. `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
.. documentation for details.

Lightweight python package for generating and analyzing graph- and matrix-based scientometrics.
This package supports loading realworld graphs as well as generating synthetic graphs to compute different metrics of interest.


Quick start
-----------

Install the package in editable mode and build the docs:

.. code-block:: powershell

    pip install git+https://github.com/Liberata-Academic-Publishing/liberata-scientometrics
    python test_scripts/matrix_generators_test.py #(this will create the appropriate matrices needed for tests)
    python test_scripts/portfolio_metrics_test.py #(change the BASE_DIR to point to the right location, and run tests for computing metrics)

.. For building docs
.. pip install -e .
.. cd docs
.. .\make.bat html
Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: API & Modules

   liberata_metrics
   api/index

Additional resources
--------------------

- `Project README <https://github.com/Liberata-Academic-Publishing/liberata-scientometrics/blob/master/README.md>`_
- Source code: Code for the package can be found `here <https://github.com/Liberata-Academic-Publishing/liberata-scientometrics>`_.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`