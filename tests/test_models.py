"""Tests for LLM regression modelling."""
from re import escape
from unittest.mock import DEFAULT, Mock, patch

from numpy import array, nan
from numpy.testing import assert_array_equal
from pandas import DataFrame
from pytest import LogCaptureFixture, raises

from llm_regression.models import OpenAiRegressor


def test_OpeanAiRegressor__repr__():
    with patch.multiple("llm_regression.models", load_dotenv=Mock, OpenAI=Mock):
        model = OpenAiRegressor()
        assert repr(model) == "OpenAiRegressor(model=gpt-3.5-turbo)"


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


def test_OpenAiRegressor_predict_returns_predictions():

    def make_mock_api_response(content: str | None) -> Mock:
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = content
        return mock_response

    with patch.multiple(
        "llm_regression.models", load_dotenv=DEFAULT, OpenAI=DEFAULT
    ) as mock_objs:
        mock_client = mock_objs["OpenAI"].return_value
        mock_client.chat.completions.create.side_effect = [
            make_mock_api_response("Output: 1.0"),
            make_mock_api_response("Output: -1.0"),
            make_mock_api_response(None)
        ]
        model = OpenAiRegressor()
        model._prompt_train_data = "Predict some stuff."
        y_pred = model.predict(array([[1.0], [0.1], [0.0]]))
        assert_array_equal(y_pred, array([[1.0], [-1.0], [nan]]))


def test_OpenAiRegressor_predict_handles_response_errors():
    with patch.multiple(
        "llm_regression.models", load_dotenv=DEFAULT, OpenAI=DEFAULT
    ) as mock_objs:
        mock_client = mock_objs["OpenAI"].return_value
        mock_client.chat.completions.create.side_effect = [Exception, Exception]
        model = OpenAiRegressor()
        model._prompt_train_data = "Predict some stuff."
        y_pred = model.predict(array([[1.0], [0.1]]))
        assert_array_equal(y_pred, array([[nan], [nan]]))


def test_OpenAiRegressor_predict_logs_errors(caplog: LogCaptureFixture):
    with patch.multiple(
        "llm_regression.models", load_dotenv=DEFAULT, OpenAI=DEFAULT
    ) as mock_objs:
        mock_client = mock_objs["OpenAI"].return_value
        mock_client.chat.completions.create.side_effect = Exception("foo")
        model = OpenAiRegressor()
        model._prompt_train_data = "Predict some stuff."
        model.predict(array([[1.0]]))

        log_record_one = caplog.records[0]
        assert len(caplog.records) == 1
        assert log_record_one.levelname == "WARNING"
        assert log_record_one.message == "LLM error for test data row #0 - foo"

        # make sure we can switch logging off
        model.predict(array([[1.0]]), logging=False)
        assert len(caplog.records) == 1


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
