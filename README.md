# liberata-scientometrics

A comprehensive Python package for computing different metrics in the Liberata System.

### Version 0.0.1

In development module for Liberata metrics compuation functions, currently written under the assumption that the capital matrix in COO, plus the ID-index mapping for contributors and manuscripts, are available as input.

### Install & Use
We highly recommend using a virtual env, either `conda` or `uv`, or any of your choice. Example showing conda environment creation

```bash
conda create -n liberata python=3.11
```

#### Clone Locally
Use `git clone` to clone the repository locally at your desired location.
```bash
conda activate liberata
cd liberata-scientometrics
pip install -r requirements.txt
pip install -e .
python test_scripts/matrix_generators_test.py #(this will create the appropriate matrices needed for tests)
python test_scripts/portfolio_metrics_test.py #(change the BASE_DIR to point to the right location, and run tests for computing metrics)
```

#### Direct install
You can also directly install the module using `git+https`

```bash
conda activate liberata
pip install git+https://github.com/Liberata-Academic-Publishing/liberata-scientometrics
python test_scripts/matrix_generators_test.py #(this will create the appropriate matrices needed for tests)
python test_scripts/portfolio_metrics_test.py #(change the BASE_DIR to point to the right location, and run tests for computing metrics)
```


## Local Testing
For local testing, navigate to the ``test_scripts/`` directory and run

### Toy Data Generation
Liberata classifies manuscript topics based on [OpenAlex](https://help.openalex.org/hc/en-us/articles/24736129405719-Topics) topics. You can download the list of topics to generate synthetic graphs based on a sample of OpenAlex topics (check `data/download_data.txt`), or, just use generic topic names (default). To run local tests, change or add your configuration in `test-scripts/config/matrix_config.yaml`  
```
python test_scripts/matrix_generators_test.py
```

The outputted matrices are stored in COO in the ``test_scripts/output/`` directory. 

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
