import argparse

import joblib
import pandas as pd
import numpy as np
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

def get_df_per_location(csv_fn: str) -> dict:
    full_df = pd.read_csv(csv_fn)
    unique_locations_list = full_df['location'].unique()
    locations = {location: full_df[full_df['location'] == location] for location in unique_locations_list}
    return locations

def predict(model_fn, historic_data_fn, future_climatedata_fn, predictions_fn):
    number_of_weeks_pred = 6
    models = joblib.load(model_fn)

    locations_future = get_df_per_location(future_climatedata_fn)
    locations_historic = get_df_per_location(historic_data_fn)

    first_location = True


    for location, df in locations_future.items():
        data = locations_historic[location]
        data = fill_disease_data(data)

        model = models[location]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['disease_cases'].values.reshape(-1, 1))

        window_size = min(313, int(len(data) / 1.5))
        #print("Window_size: ", window_size)

        X = scaled_data[-window_size:].flatten()

        #print("X: ", X)

        predictions = []

        for i in range(number_of_weeks_pred):
            X_reshaped = X.reshape(1, window_size, 1)
            prediction_scaled = model.predict(X_reshaped, verbose=0)[0][0]
            prediction = scaler.inverse_transform([[prediction_scaled]])[0][0]
            predictions.append(prediction)
            X = np.append(X[1:], prediction_scaled)

        df['sample_0'] = np.array(predictions)

        if first_location:
            df.to_csv(predictions_fn, index=False, mode='w', header=True)
            first_location = False
        else:
            df.to_csv(predictions_fn, index=False, mode='a', header=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict using the trained model.')

    parser.add_argument('model_fn', type=str, help='Path to the trained model file.')
    parser.add_argument('historic_data_fn', type=str, help='Path to the CSV file historic data (here ignored).')
    parser.add_argument('future_climatedata_fn', type=str, help='Path to the CSV file containing future climate data.')
    parser.add_argument('predictions_fn', type=str, help='Path to save the predictions CSV file.')

    args = parser.parse_args()
    predict(args.model_fn, args.historic_data_fn, args.future_climatedata_fn, args.predictions_fn)
