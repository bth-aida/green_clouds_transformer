import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm

from data_processing import add_features_tree, load_data


def mue(pred: pd.Series, real: pd.Series):
    """Calculates mean under error"""
    under_errors = real - pred
    return np.mean(under_errors[under_errors > 0])


def upf(pred: pd.Series, real: pd.Series):
    """Calculates under prediction frequency"""
    real = np.array(real)  # type: ignore
    pred = np.array(pred)  # type: ignore
    return np.mean(pred < real)


def run_xgb_grid_search(
    df_train: pd.DataFrame, df_test: pd.DataFrame, param_distr: dict
):
    X_train = df_train.drop(columns=["timestamp", "target", "id", "id_part"])
    y_train = df_train["target"]
    X_test = df_test.drop(columns=["target", "timestamp", "id", "id_part"])

    base_model = xgb.XGBRegressor(enable_categorical=True, random_state=42)
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distr,
        n_iter=15,
        cv=5,
        scoring="neg_root_mean_squared_error",
        random_state=42,
    )

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    y_pred = best_model.predict(X_test)  # type: ignore
    return y_pred, random_search.best_params_


def run_xgb():
    # load data
    _, df_train = load_data(size=10, clean=True)
    _, df_test = load_data(size=5, start_index=10, clean=True)

    # add features
    df_train = add_features_tree(df_train)
    df_test = add_features_tree(df_test)

    # run xgb model
    df_results = df_test.copy()
    df_results = df_results[["timestamp", "id", "id_part", "cpu_usage"]]

    # train and test
    param_distributions = {
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "n_estimators": [50, 100, 200],
        "subsample": [0.6, 0.8, 1.0],
    }

    for hz in tqdm(range(1, 13), "XGBoost Hz"):
        # set target
        df_hz_train = df_train.copy()
        df_hz_train["target"] = df_hz_train.groupby("id_part")["cpu_usage"].shift(-hz)
        df_hz_train.dropna(subset=["target"], inplace=True)

        df_hz_test = df_test.copy()
        df_hz_test["target"] = df_hz_test.groupby("id_part")["cpu_usage"].shift(-hz)
        df_hz_test.dropna(subset=["target"], inplace=True)

        # run model
        y_pred, _ = run_xgb_grid_search(df_hz_train, df_hz_test, param_distributions)

        df_pred = pd.DataFrame(
            {
                "timestamp": df_hz_test["timestamp"] + pd.Timedelta(minutes=5 * hz),  # type: ignore
                "id": df_hz_test["id"],
                "pred": y_pred,
            }
        )
        merged = pd.merge(df_pred, df_test, on=["timestamp", "id"], how="inner")
        merged = merged.dropna(subset=["pred", "cpu_usage"])

        y_true = merged["cpu_usage"]
        y_hat = merged["pred"]

        rmse = np.sqrt(mean_squared_error(y_true, y_hat))
        mae = mean_absolute_error(y_true, y_hat)
        mue_val = mue(y_hat, y_true)
        upf_val = upf(y_hat, y_true)

        # results
        print(f"--- Results for hz {hz} ---")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"MUE:  {mue_val:.4f}")
        print(f"UPF:  {upf_val:.4f}")


def main():
    run_xgb()


if __name__ == "__main__":
    main()
