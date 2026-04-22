# liberata-metrics

A comprehensive Python package for computing different metrics in the Liberata System.

### Version 0.0.1

In development module for Liberata metrics compuation functions, currently written under the assumption that the capital matrix in COO, plus the ID-index mapping for contributors and manuscripts, are available as input.

### Install & Use

```bash
cd liberata-metrics
pip install -e .
python test_scripts/matrix_generators_test.py
python test_scripts/portfolio_metrics_test.py #(need to change the BASE_DIR to point to the right location)
```


## Local Testing
For local testing, navigate to the ``examples/`` directory and run

### Toy Data Generation

```
python generate_matrix.py
```

The outputted matrices are stored in COO in the ``examples/output/`` directory. To customize the configuration of matrix generation, modify the YAML config file at ``examples/config/matrix_config.yaml``.  

### Portfolio Metrics Function tests
Update the path names in ``test_scripts/portfolio_metrics_test.py`` to point to the appropriate directory storing the toy matrices. Then from project root, run
```
python test_scripts/portfolio_metrics_test.py
```


### Build Documentation
```
sphinx-apidoc -o docs/source/api src/liberata_metrics -f --separate

sphinx-build -b html docs/source docs/build/html

sphinx-autobuild docs/source docs/build/html # Open http://127.0.0.1:8000 in browser
```