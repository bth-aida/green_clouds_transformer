import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)


def create_report(
    df_results: pd.DataFrame,
    duration: float,
    run_name: str,
    training_size: int,
):

    df_results.to_csv(f"results_{run_name}.csv")

    # # nan filter
    # mask = ~np.isnan(y_pred)
    # num_dropped = np.sum(~mask)
    # num_used = np.sum(mask)

    # mae = mean_absolute_error(df_test["target"][mask], y_pred[mask])
    # rmse = root_mean_squared_error(df_test["target"][mask], y_pred[mask])

    # report_name = f"report {run_name}.txt"
    # with open(report_name, "w") as f:
    #     f.write(f"{run_name}\n")
    #     f.write(f"Duration: {duration}\n")
    #     f.write(f"Training size: {training_size}\n")
    #     f.write(f"Eval size: {num_used} (dropped: {num_dropped})\n")
    #     f.write(f"Hz: {hz}\n")
    #     f.write(f"MAE: {mae}\n")
    #     f.write(f"RMSE: {rmse}\n")
