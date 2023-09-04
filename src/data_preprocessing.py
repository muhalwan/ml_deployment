# src/data_preprocessing.py

import os
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def fetch_and_save_raw_data(filepath):
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    df.to_csv(filepath, index=False)

def preprocess_and_save_data(raw_filepath, processed_filepath):
    df = pd.read_csv(raw_filepath)

    # Splitting the data
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

    # Saving the processed data
    train_set.to_csv(processed_filepath, index=False)

def main():
    base_dir = os.path.dirname(os.path.realpath(__file__))
    raw_data_filepath = os.path.join(base_dir, '..', 'data', 'raw', 'california_housing_raw.csv')
    processed_data_filepath = os.path.join(base_dir, '..', 'data', 'processed', 'california_housing_processed.csv')

    if not os.path.exists(raw_data_filepath):
        fetch_and_save_raw_data(raw_data_filepath)

    preprocess_and_save_data(raw_data_filepath, processed_data_filepath)

if __name__ == "__main__":
    main()
