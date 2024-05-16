"""Tests for utlity code."""
from numpy import corrcoef
from pytest import approx

from llm_regression import make_univariate_linear_test_data


def test_make_univariate_linear_test_data():
    corr_coeff = 0.8
    dataset = make_univariate_linear_test_data(100000, rho=corr_coeff)
    assert "x" in dataset.columns
    assert "y" in dataset.columns
    assert dataset["x"].mean() == approx(0.0, abs=0.01)
    assert dataset["x"].std() == approx(1.0, abs=0.01)
    assert dataset["y"].mean() == approx(0.0, abs=0.01)
    assert dataset["y"].std() == approx(1.0, abs=0.01)
    assert corrcoef(dataset["x"], dataset["y"])[0][1] == approx(corr_coeff, abs=0.01)
