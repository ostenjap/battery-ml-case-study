"""ARIMA baseline for capacity fade, for comparison against the PINN."""
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from data_loader import load_nasa_capacity_fade, train_test_split_by_cycle, processed_csvs


def fit_and_forecast(csv_path, order=(2, 1, 2)):
    df = load_nasa_capacity_fade(csv_path)
    train_df, test_df = train_test_split_by_cycle(df)

    model = ARIMA(train_df["capacity_ah"].to_numpy(), order=order)
    fit = model.fit()
    forecast = fit.forecast(steps=len(test_df))

    return forecast, test_df["capacity_ah"].to_numpy(), train_df, test_df


if __name__ == "__main__":
    csv_files = processed_csvs()
    if not csv_files:
        print("No per-cell CSVs yet. Run: python src/explore_nasa.py")
    else:
        forecast, actual, train_df, test_df = fit_and_forecast(csv_files[0])
        print("ARIMA forecast vs actual (first 10):")
        print(np.column_stack([forecast[:10], actual[:10]]))
