# Contributing to Liberata Scientometrics

Thanks for helping improve the library. Contributions are welcome as bug reports, metric corrections, tests, documentation, and pull requests.

The package lives in `src/liberata_metrics`. Tests live in `test_scripts/`.

## Development setup

Use Python 3.11. The package declares `requires-python >= 3.8`, and the documentation workflow runs on 3.11.

```bash
git clone https://github.com/Liberata-Academic-Publishing/liberata-scientometrics
cd liberata-scientometrics
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For work on `src/liberata_metrics/integrations` or `supabase_test_scripts/`, install the optional extra as well:

```bash
pip install -e ".[supabase]"
```

Do not commit `.env` files, API keys, or generated matrices. See [What not to commit](#what-not-to-commit).

## Tests

The default suite uses small in-memory fixtures and does not need generated data:

```bash
pytest -m "not large"
```

Tests marked `large` read pre-generated matrices from disk. `test_scripts/conftest.py` points `LARGE_DATA_DIR` at a folder under `test_scripts/output/`. If that folder is missing, those tests are skipped.

To run the large suite:

1. Generate matrices (see below), or place an existing output folder under `test_scripts/output/`.
2. Set `LARGE_DATA_DIR` in `test_scripts/conftest.py` to that folder.
3. Run `pytest -m large`.

A change to a metric should include a test in the matching file under `test_scripts/`. Prefer the small fixtures in `conftest.py` unless the behavior only shows up at scale.

## Synthetic data

Topic settings live in `test_scripts/config/matrix_config.yaml`. Generate a local dataset with:

```bash
python test_scripts/matrix_generators_test.py
```

Output is written to `test_scripts/output/` in COO format. That directory is gitignored.

## Documentation

API pages are generated from docstrings. After changing public functions, rebuild the docs:

```bash
pip install sphinx pydata-sphinx-theme myst-parser sphinx-autobuild
sphinx-apidoc -o docs/source/api/generated src/liberata_metrics -f --separate
sphinx-build -b html docs/source docs/build/html
```

For a local preview:

```bash
sphinx-autobuild docs/source docs/build/html
```

That serves the docs at `http://127.0.0.1:8000`. The published site is [liberata.info](https://liberata.info).

## Pull requests

- Open an issue first for a new metric or a change to an existing formula, so the definition can be agreed on before the code.
- Keep the pull request focused on one change.
- Match the style of the module you are editing. There is no separate formatter configured.
- Describe what changed and how you checked it. Include the `pytest` command you ran.
- New or changed public functions need a docstring. The docs are built from those docstrings.

By submitting a pull request, you agree that your contribution is licensed under the Apache License 2.0, the same license as the rest of the repository.

## Reporting issues

Use [GitHub Issues](https://github.com/Liberata-Academic-Publishing/liberata-scientometrics/issues) for bugs and metric questions. Include:

- the metric or module
- a minimal input (shapes of `references`, `shares`, or `capital`, plus a seed if you used the generators)
- the result you got and the result you expected

## What not to commit

These paths are gitignored and should stay that way:

- `.env` and other credential files
- `test_scripts/output/`
- `data/*.csv`, including the OpenAlex topic mapping table
- `docs/build/`

If a test needs a credential or a large matrix, document how to obtain it in the pull request. Do not paste secrets into issues or commits.
