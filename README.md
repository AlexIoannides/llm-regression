# Regression using LLMs

The llm-regression package demonstrates how LLMs can be used to solve classical regression problems, and exposes these capabilities for you to experiment with. Example:

```python
from llm_regression import OpenAiRegressor

llm_regressor = OpenAiRegressor(model="gpt-3.5-turbo")
llm_regressor.fit(X_train, y_train)
y_pred = llm_regressor.predict(X_test)
```

This work was motivated by the paper,

["_From Words to Numbers: You LLM is Secretly a Capable Regressor_", by Vacareanu et al. (2024)](https://arxiv.org/abs/2404.07544).

Which is well worth a read!

## Installing

You can install the llm_regression package, together with the dependencies required to run the example notebooks, directly from this repo,

```text
pip install -U pip
pip install "llm-regression[examples] @ git+https://github.com/AlexIoannides/llm-regression.git"
```

## Examples

Checkout the [basic_demo notebook](https://github.com/AlexIoannides/llm-regression/tree/main/examples/basic_demo.ipynb).

## Developer Setup

If you want to modify or extend the work in this repo, then the information in this section is for you.

### Install Developer Tools

Install the package as an [editable dependency](https://setuptools.pypa.io/en/latest/userguide/development_mode.html), together with all the developer tools required to format code, check types and run tests:

```text
pip install -e ".[dev]"
```

### Developer Task Execution with Nox

We use [Nox](https://nox.thea.codes/en/stable/) for scripting developer tasks, such as formatting code, checking types and running tests. These tasks are defined in `noxfile.py`, a list of which can be returned on the command line,

```text
$ nox --list

Sessions defined in /Users/.../noxfile.py:

* run_tests-3.12 -> Run unit tests.
- format_code-3.12 -> Lint code and re-format where necessary.
* check_code_formatting-3.12 -> Check code for formatting errors.
* check_types-3.12 -> Run static type checking.
- build_and_deploy-3.12 -> Build wheel and deploy to PyPI.

sessions marked with * are selected, sessions marked with - are skipped.
```

Single tasks can be executed easily - e.g.,

```text
$ nox -s run_tests

nox > Running session run_tests-3.12
nox > Creating virtual environment (virtualenv) using python3.12 in .nox/run_tests-3-10
nox > python -m pip install '.[dev]'
nox > pytest 
======================================== test session starts ========================================
platform darwin -- Python 3.12.2, pytest-7.4.2, pluggy-1.3.0
rootdir: /Users/.../llm_regression
configfile: pyproject.toml
testpaths: tests
collected 1 item                                                                                                                                                         

tests/test_hello_world.py                      [100%]

========================================== 1 passed in 0.00s =========================================
nox > Session run_tests-3.12 was successful.
```

### CI/CD

This repo comes configured to run two [GitHub Actions](https://docs.github.com/en/actions) workflows:

- **Test Python Package (CI)**, defined in `.github/workflows/python-package-ci.yml`
- **Deploy Python Package (CD)**, defined in `.github/workflows/python-package-cd.yml`

The CI workflow has been configured to run whenever a pull request to the `main` branch is created. The CD workflow has been configured to run whenever a [release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) is created on GitHub.
