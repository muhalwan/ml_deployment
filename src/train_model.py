# src/train_model.py

import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np


def load_processed_data(filepath):
    return pd.read_csv(filepath)

def train_linear_regression(df):
    # Features and target variable
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    # Training the model
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)

    # Calculate the RMSE for evaluation (optional)
    predictions = lin_reg.predict(X)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    print(f"Training RMSE: {rmse}")

    return lin_reg

def save_model(model, filepath):
    joblib.dump(model, filepath)

def main():
    base_dir = os.path.dirname(os.path.realpath(__file__))
    processed_data_filepath = os.path.join(base_dir, '..', 'data', 'processed', 'california_housing_processed.csv')
    model_filepath = os.path.join(base_dir, '..', 'models', 'linear_regression_model.pkl')

    df = load_processed_data(processed_data_filepath)
    model = train_linear_regression(df)
    save_model(model, model_filepath)

if __name__ == "__main__":
    main()
