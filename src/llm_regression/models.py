"""Regression models using LLMs."""
import re

import numpy as np
from dotenv import load_dotenv
from numpy.random import default_rng
from openai import BadRequestError, OpenAI
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def predict(
    test_data: DataFrame, train_data: DataFrame, *, verbose: bool = False
) -> DataFrame:
    """Score a dataset using an LLM.

    Args:
    ----
        test_data: Dataframe of features/variables to use for prediction.
        train_data: Dataframe of labelled features/variables to use for training.
        verbose: Print prompt for first test data instances?

    Returns:
    -------
        A dataframe with predicted values for the test data.
    """
    load_dotenv()  # load OPEN_API_KEY from .env file (if present)
    client = OpenAI()

    system_prompt = (
        "Your task is to provide your best estimate for ”Output”. Please provide that "
        "and only that, without any additional text."
    )

    prompt_train_data = [
        f"Feature 0: {row.x}\nOutput: {row.y}" for row in train_data.itertuples()
    ]

    y_pred: list[float] = []
    for row in tqdm(
        test_data.itertuples(),
        total=test_data.shape[0],
    ):
        prompt_test_data = [f"Feature 0: {row.x}\nOutput:"]

        user_prompt = "\n\n".join(prompt_train_data + prompt_test_data)
        if verbose:
            print(user_prompt)

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "text"},
            seed=42,
        )
        try:
            prediction = completion.choices[0].message.content
        except BadRequestError as e:
            raise ModelError("API call to LLM failed") from e
        if prediction:
            y_pred += [_parse_model_output(prediction)]
        else:
            raise ModelError("prediciton failed")

    return DataFrame({"y_pred": y_pred})


def _parse_model_output(output: str) -> float:
    """Parse the models's output."""
    try:
        result = re.findall(r"-?\d+\.?\d*", output)[0]
        return float(result)
    except (ValueError, IndexError) as e:
        raise ModelError("invalid model prediction") from e


def make_univariate_linear_test_data(
    n_samples: int = 1000, *, rho: float = 0.75, seed: int = 42
) -> DataFrame:
    """Simulate a y = rho * x + sqrt(1 - rho ** 2) * epsilon.

    Args:
    ----
        n_samples: Number of samples to generate. Defaults to 1000.
        rho: Rho coeffcient (correlation coefficient). Defaults to 0.75.
        seed: Random seed. Defaults to 42.

    Returns:
    -------
        Dataframe of test data.
    """
    if not (rho >= 0 and rho <= 1):
        raise ValueError(f"rho = {rho} - must in [0, 1]")
    rng = default_rng(seed)
    x = rng.standard_normal(n_samples)
    epsilon = rng.standard_normal(n_samples)
    y = rho * x + np.sqrt(1 - rho * rho) * epsilon
    return DataFrame({"x": x, "y": y})


class ModelError(Exception):
    """Custom exception class for model errors."""

    pass


if __name__ == "__main__":
    # make datasets
    n_samples = 1000
    dataset = make_univariate_linear_test_data(n_samples, rho=0.9)
    train_data, test_data = train_test_split(dataset, test_size=0.05, random_state=42)

    # ols regression
    ols_regressor = LinearRegression()
    ols_regressor.fit(train_data[["x"]], train_data[["y"]])
    y_pred_ols = ols_regressor.predict(test_data[["x"]])

    ols_results = test_data.copy().reset_index(drop=True).assign(y_pred=y_pred_ols)
    mean_abs_err_ols = mean_absolute_error(ols_results["y"], ols_results["y_pred"])
    r_squared_ols = r2_score(ols_results["y"], ols_results["y_pred"])
    print(f"mean_abs_error = {mean_abs_err_ols}")
    print(f"r_squared = {r_squared_ols}")

    # llm regression
    y_pred = predict(test_data, train_data)

    llm_results = (
        test_data.copy().reset_index(drop=True).assign(y_pred=y_pred["y_pred"])
    )
    mean_abs_err_llm = mean_absolute_error(llm_results["y"], llm_results["y_pred"])
    r_squared_llm = r2_score(llm_results["y"], llm_results["y_pred"])
    print(f"mean_abs_error = {mean_abs_err_llm}")
    print(f"r_squared = {r_squared_llm}")

# mean_abs_error = 0.4107320869725583
# r_squared = 0.7865828324377897
# mean_abs_error = 0.38392287248603985
# r_squared = 0.8083485333779725
