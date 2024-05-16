"""Regression modelling using LLMs."""
from __future__ import annotations

import re
from logging import getLogger
from typing import Literal

import numpy as np
from dotenv import load_dotenv
from numpy import ndarray
from openai import OpenAI
from pandas import DataFrame
from tqdm import tqdm

OpenAiModel = Literal["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo", "gpt-4"]

log = getLogger("OpenAIRegressionLogger")


class OpenAiRegressor:
    """Generic regression using Open AI LLMs."""

    def __init__(self, model: OpenAiModel = "gpt-3.5-turbo", seed: int = 42):
        """Initialise object.

        Args:
        ----
            model: Open AI model to use. Defaults to "gpt-3.5-turbo".
            seed: Random seed to use with OpenAI model. Defaults to 42
        """
        load_dotenv()  # load OPEN_API_KEY from .env file (if present)
        self._client = OpenAI()
        self._model = model
        self._model_seed = seed
        self._prompt_instruction = (
            "Your task is to provide your best estimate for ”Output”. Please provide "
            "that and only that, without any additional text."
        )
        self._prompt_train_data: str = ""

    def __repr__(self) -> str:
        """Create string representation."""
        return f"OpenAiRegressor(model={self._model})"

    def fit(self, X: DataFrame | ndarray, y: DataFrame | ndarray) -> OpenAiRegressor:
        """Create a prompt based on training data to use when predicting with an LLM.

        Args:
        ----
            X: Feature data.
            y: Labels.

        Raises:
        ------
            ValueError: If the dimensions of X or y are invalid and/or inconsistent with
                one another.

        Returns:
        -------
            The OpenAiRegressor object.
        """
        if X.ndim < 2:
            raise ValueError("X.ndim must be >= 2")
        if y.ndim < 2:
            raise ValueError("y.ndim must be == 2")
        if len(X) != len(y):
            raise ValueError("len(y) != len(X)")

        _X = X.tolist() if isinstance(X, ndarray) else X.values.tolist()
        _y = y.tolist() if isinstance(y, ndarray) else y.values.tolist()

        self._prompt_train_data = "\n\n".join(
            [self._format_data_row(row, _y[n_row]) for n_row, row in enumerate(_X)]
        )

        return self

    def predict(self, X: DataFrame | ndarray, logging: bool = True) -> ndarray:
        """Predict labels using model and feature data.

        Any prediction failures will return `numpy.nan` - prediction won't be halted,
        given the expense of querying LLMs.

        Args:
        ----
            X: Feature data to use for predictions.
            logging: Enable logging. Default to True.

        Raises:
        ------
            RuntimeError: If `.fit` has not been called.

        Returns:
        -------
            Model predictions
        """
        if not self._prompt_train_data:
            raise RuntimeError("please fit model before trying to generate predictions")

        _X = X if isinstance(X, ndarray) else X.values
        y_pred: list[float] = []

        for n, row in tqdm(enumerate(_X), total=len(_X)):
            try:
                prediction_prompt = self._compose_prediction_prompt(
                    self._prompt_instruction,
                    self._prompt_train_data,
                    self._format_data_row(row),
                )
                llm_response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prediction_prompt}],
                    temperature=0,
                    response_format={"type": "text"},
                    seed=self._model_seed,
                )
                llm_generation = llm_response.choices[0].message.content
                if llm_generation:
                    y_pred += [self._parse_model_output(llm_generation)]
                else:
                    y_pred += [np.nan]
            except Exception as e:
                if logging:
                    log.warning(f"LLM error for test data row #{n} - {str(e)}")
                y_pred += [np.nan]

        return np.array(y_pred).reshape(-1, 1)

    @staticmethod
    def _compose_prediction_prompt(
        instruction: str, train_data: str, test_data: str
    ) -> str:
        """Compose full prompt from constituent parts."""
        return instruction + "\n" + train_data + "\n\n" + test_data

    @staticmethod
    def _format_data_row(x_row: ndarray, y_row: ndarray | None = None) -> str:
        """Format a data row for inclusion in model prompt."""
        output = y_row[0] if y_row else ""
        prompt_data = "\n".join(
            [f"Feature {n}: {x}" for n, x in enumerate(x_row)] + [f"Output: {output}"]
        )
        return prompt_data

    @staticmethod
    def _parse_model_output(output: str) -> float:
        """Parse the models's output."""
        result = re.findall(r"-?\d+\.?\d*", output)[0]
        return float(result)
