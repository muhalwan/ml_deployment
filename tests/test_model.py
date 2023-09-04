# tests/test_model.py

import pytest
from model import load_data, preprocess_data, train_model, predict

def test_load_data():
    data = load_data()
    assert isinstance(data, list), f"Expected data to be of type list, but got {type(data)}"
    assert len(data) > 0, "Expected data to have some values"

def test_preprocess_data():
    data = [1, 2, 3, 4, 5]
    processed_data = preprocess_data(data)
    assert processed_data.shape == (5,), f"Expected shape (5,) but got {processed_data.shape}"

def test_train_model():
    data = [1, 2, 3, 4, 5]
    model = train_model(data)
    assert model == 3, f"Expected model (mean) value to be 3, but got {model}"

def test_predict():
    model = 3
    new_data = [10]
    prediction = predict(model, new_data)
    assert prediction == 3, f"Expected prediction to be 3, but got {prediction}"
