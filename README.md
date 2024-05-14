# llm_regression

This is the repository for the llm_regression Python package.

## Developer Setup

Install the package as an [editable dependency](https://setuptools.pypa.io/en/latest/userguide/development_mode.html), together with all the developer tools required to format code, check types and run tests:

```text
$ pip install -e ".[dev]"
```

### Developer Task Execution with Nox

We use [Nox](https://nox.thea.codes/en/stable/) for scripting developer tasks, such as formatting code, checking types and running tests. These tasks are defined in `noxfile.py`, a list of which can be returned on the command line,

```text
$ nox --list

Sessions defined in /Users/.../noxfile.py:

* run_tests-3.10 -> Run unit tests.
- format_code-3.10 -> Lint code and re-format where necessary.
* check_code_formatting-3.10 -> Check code for formatting errors.
* check_types-3.10 -> Run static type checking.
- build_and_deploy-3.10 -> Build wheel and deploy to PyPI.

sessions marked with * are selected, sessions marked with - are skipped.
```

Single tasks can be executed easily - e.g.,

```text
$ nox -s run_tests

nox > Running session run_tests-3.10
nox > Creating virtual environment (virtualenv) using python3.10 in .nox/run_tests-3-10
nox > python -m pip install '.[dev]'
nox > pytest 
======================================== test session starts ========================================
platform darwin -- Python 3.10.2, pytest-7.4.2, pluggy-1.3.0
rootdir: /Users/.../llm_regression
configfile: pyproject.toml
testpaths: tests
collected 1 item                                                                                                                                                         

tests/test_hello_world.py .                                                                                                                                        [100%]

========================================== 1 passed in 0.00s =========================================
nox > Session run_tests-3.10 was successful.
```

### Building Packages and Deploying to PyPI

This is automated via the `nox -s build_and_deploy` command. In order to use this, the following environment variables will need to be made available to Python:

```text
PYPI_USR  # PyPI username
PYPI_PWD  # PyPI password
```

These may be specified in a `.env` file from which they will be loaded automatically - e.g.,

```text
PYPI_USR=XXXX
PYPI_PWD=XXXX
```

Note: `.gitignore` will ensure that `.env`is not tracked by Git.

## CI/CD

This repo comes configured to run two [GitHub Actions](https://docs.github.com/en/actions) workflows:

- **Test Python Package (CI)**, defined in `.github/workflows/python-package-ci.yml`
- **Deploy Python Package (CD)**, defined in `.github/workflows/python-package-cd.yml`

The CI workflow has been configured to run whenever a pull request to the `main` branch is created. The CD workflow has been configured to run whenever a [release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) is created on GitHub.

Note, the CD workflow will require `PYPI_USR` and `PYPI_PWD` to be added as [repository secrets](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions).
