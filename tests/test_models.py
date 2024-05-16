"""Tests for LLM regression modelling."""
from re import escape
from unittest.mock import Mock, patch

from numpy import array
from pandas import DataFrame
from pytest import raises

from llm_regression.models import OpenAiRegressor


def test_OpeanAiRegressor_fit_makes_prompt_train_data_pandas_dataframe():
    with patch.multiple("llm_regression.models", load_dotenv=Mock, OpenAI=Mock):
        train_data = DataFrame(
            {"x0": [1.0, -0.1], "x1": [0.1, -1.0], "y": [1.0, 2.0]}
        )
        X = train_data[["x0", "x1"]]
        y = train_data[["y"]]

        model = OpenAiRegressor()

        expected_prompt = (
            "Feature 0: 1.0\nFeature 1: 0.1\nOutput: 1.0\n\n"
            "Feature 0: -0.1\nFeature 1: -1.0\nOutput: 2.0"
        )

        assert model.fit(X, y)._prompt_train_data == expected_prompt


def test_OpeanAiRegressor_fit_makes_prompt_train_data_numpy_array():
    with patch.multiple("llm_regression.models", load_dotenv=Mock, OpenAI=Mock):
        X = array([[1.0, 0.1], [-0.1, -1.0]])
        y = array([[1.0], [2.0]])
        model = OpenAiRegressor()

        expected_prompt = (
            "Feature 0: 1.0\nFeature 1: 0.1\nOutput: 1.0\n\n"
            "Feature 0: -0.1\nFeature 1: -1.0\nOutput: 2.0"
        )

        assert model.fit(X, y)._prompt_train_data == expected_prompt


def test_OpeanAiRegressor_fit_raises_errors_on_inconsistent_inputs():
    with patch.multiple("llm_regression.models", load_dotenv=Mock, OpenAI=Mock):
        model = OpenAiRegressor()

        with raises(ValueError, match=escape("X.ndim must be >= 2")):
            model.fit(array([1.0, 0.1, -0.1, -1.0]), array([[1.0], [2.0]]))

        with raises(ValueError, match=escape("y.ndim must be == 2")):
            model.fit(array([[1.0, 0.1], [-0.1, -1.0]]), array([1.0, 2.0]))

        with raises(ValueError, match=escape("len(y) != len(X)")):
            model.fit(array([[1.0, 0.1], [-0.1, -1.0]]), array([[1.0]]))


def test_OpeanAiRegressor__repr__():
    with patch.multiple("llm_regression.models", load_dotenv=Mock, OpenAI=Mock):
        model = OpenAiRegressor()
        assert repr(model) == "OpenAiRegressor(model=gpt-3.5-turbo)"


def test_OpeanAiRegressor_compose_prediction_prompt():
    assert OpenAiRegressor._compose_prediction_prompt("a", "b", "c") == "a\nb\n\nc"


def test_OpeanAiRegressor_format_data_row():
    assert OpenAiRegressor._format_data_row([1.0]) == "Feature 0: 1.0\nOutput: "
    assert OpenAiRegressor._format_data_row([1.0, -0.1]) == "Feature 0: 1.0\nFeature 1: -0.1\nOutput: "  # noqa
    assert OpenAiRegressor._format_data_row([1.0], [0.1]) == "Feature 0: 1.0\nOutput: 0.1"  # noqa


def test_OpenAiRegressor_parse_model_output():
    assert OpenAiRegressor._parse_model_output("\nOutput: 0.101") == 0.101
    assert OpenAiRegressor._parse_model_output("Output: -1.101") == -1.101
    assert OpenAiRegressor._parse_model_output("Output: 1") == 1.0
