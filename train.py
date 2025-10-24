import argparse

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def fill_disease_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN values in disease case data by:
    1. Forward filling,
    2. Backward filling,
    3. Filling remaining NaNs with 0.
    """
    return df.ffill().bfill().fillna(0)


def prepare_sequences(data, window_size):
    """ Prepare sequences for LSTM training """
    X, y = [], []
    for i in range(len(data) - window_size):
        seq_x = data[i:(i + window_size)]
        seq_y = data[i + window_size]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y).reshape(-1, 1)


def get_df_per_location(csv_fn: str) -> dict:
    full_df = pd.read_csv(csv_fn)
    unique_locations_list = full_df['location'].unique()
    locations = {location: full_df[full_df['location'] == location] for location in unique_locations_list}
    return locations

def train(csv_fn, model_fn, model_config):
    models = {}
    locations = get_df_per_location(csv_fn)
    num_units = 1
    num_epochs = 1

    for location, data in locations.items():
        data = fill_disease_data(data)
        window_size = int(len(data) * 0.5)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['disease_cases'].values.reshape(-1, 1))


        X, y = prepare_sequences(scaled_data, window_size)

        # Define LSTM model
        model = Sequential([
            LSTM(num_units, activation='relu', input_shape=(window_size, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Fit the LSTM model
        model.fit(X, y, epochs=num_epochs, verbose=1)

        models[location] = model

    joblib.dump(models, model_fn)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a minimalist forecasting model.')

    parser.add_argument('csv_fn', type=str, help='Path to the CSV file containing input data.')
    parser.add_argument('model_fn', type=str, help='Path to save the trained model.')
    parser.add_argument('model_config', type=str, help='Model_configurations')
    args = parser.parse_args()
    train(args.csv_fn, args.model_fn, args.model_config)


