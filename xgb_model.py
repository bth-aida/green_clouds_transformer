import time

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm

from data_processing import add_features_tree, load_data


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
    _, df_train = load_data(size=100, clean=True)
    _, df_test = load_data(size=50, start_index=100, clean=True)

    # add features
    df_train = add_features_tree(df_train)
    df_test = add_features_tree(df_test)

    # run xgb model
    df_results = df_test.copy()
    df_results = df_results[["timestamp", "id", "id_part", "cpu_usage"]]

    # train and test
    start_time = time.time()

    param_distributions = {
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "n_estimators": [50, 100, 200],
        "subsample": [0.6, 0.8, 1.0],
    }

    best_params = []

    for hz in tqdm(range(1, 13), "XGBoost Hz"):
        # set target
        df_hz_train = df_train.copy()
        df_hz_train["target"] = df_hz_train.groupby("id_part")["cpu_usage"].shift(-hz)
        df_hz_train.dropna(subset=["target"], inplace=True)

        df_hz_test = df_test.copy()
        df_hz_test["target"] = df_hz_test.groupby("id_part")["cpu_usage"].shift(-hz)
        df_hz_test.dropna(subset=["target"], inplace=True)

        # run model
        y_pred, params = run_xgb_grid_search(
            df_hz_train, df_hz_test, param_distributions
        )

        # store best params
        best_params.append((hz, params))

        df_pred = pd.DataFrame(
            {
                "timestamp": df_hz_test["timestamp"] + pd.Timedelta(minutes=5 * hz),  # type: ignore
                "id": df_hz_test["id"],
                f"pred_{hz}": y_pred,
            }
        )

        df_results = df_results.merge(df_pred, on=["timestamp", "id"], how="left")

    duration = time.time() - start_time
    print("Duration: ", duration)
    for bp in best_params:
        print(bp)
    df_results.to_csv("results_xgb.csv")

    # TODO: save duration and best params in report


def main():
    run_xgb()


if __name__ == "__main__":
    main()
