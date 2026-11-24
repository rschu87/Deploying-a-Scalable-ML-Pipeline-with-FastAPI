import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def data():
    """Load a fixed sample of the census data to keep the tests fast."""
    df = pd.read_csv(os.path.join(PROJECT_PATH, "data", "census.csv"))
    return df.sample(n=1000, random_state=42)


@pytest.fixture(scope="module")
def processed(data):
    """Process the sample the same way train_model.py does."""
    X, y, encoder, lb = process_data(
        data, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    return X, y


def test_process_data_shapes(data, processed):
    """
    process_data should return one row per input row, numeric features only,
    and a binary label array of the same length.
    """
    X, y = processed
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == len(data)
    assert y.shape[0] == len(data)
    assert set(np.unique(y)) <= {0, 1}


def test_train_model_returns_random_forest(processed):
    """
    train_model should return a fitted RandomForestClassifier.
    """
    X, y = processed
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "classes_")


def test_inference_returns_binary_predictions(processed):
    """
    inference should return one 0/1 prediction per input row.
    """
    X, y = processed
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]
    assert set(np.unique(preds)) <= {0, 1}


def test_compute_model_metrics_known_values():
    """
    compute_model_metrics should return the expected precision, recall and F1
    on a small hand-checked example: 2 true positives, 1 false positive and
    1 false negative gives precision 2/3, recall 2/3 and F1 2/3.
    """
    y = np.array([1, 1, 1, 0, 0])
    preds = np.array([1, 1, 0, 1, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)
    assert fbeta == pytest.approx(2 / 3)
