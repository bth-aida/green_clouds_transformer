import numpy as np
import pandas as pd


def mue(pred: pd.Series, real: pd.Series):
    """Calculated mean under error"""
    under_errors = real - pred
    return np.mean(under_errors[under_errors > 0])


def upf(pred: pd.Series, real: pd.Series):
    """Calculates under prediction frequency"""
    real = np.array(real)  # type: ignore
    pred = np.array(pred)  # type: ignore
    return np.mean(pred < real)


def main():
    # load data
    df_res = pd.read_csv("df_res.csv")
    pred_tf = pd.read_csv("pred_tf.csv")

    pred_tf_selected = pred_tf[["idx", "timestamp", "pred_11"]].rename(
        columns={"pred_11": "pred_11_tf"}
    )

    df_res_selected = df_res[["idx", "timestamp", "target", "pred_11"]].rename(
        columns={"pred_11": "pred_11_xgb"}
    )

    merged_df = pd.merge(
        df_res_selected, pred_tf_selected, on=["timestamp", "idx"], how="inner"
    )
    final_df = merged_df[["timestamp", "idx", "target", "pred_11_xgb", "pred_11_tf"]]

    # filter out
    # final_df = final_df[final_df["target"] > 0.05]
    # final_df.to_csv("dff.csv")

    # XGBoost
    target = final_df["target"]
    pred_xgb = final_df["pred_11_xgb"]

    rmse_xgb = np.sqrt(((target - pred_xgb) ** 2).mean())
    mae_xgb = (target - pred_xgb).abs().mean()
    mue_xgb = mue(target, pred_xgb)
    upf_xgb = upf(target, pred_xgb)

    # Tf
    pred_tf = final_df["pred_11_tf"]

    rmse_tf = np.sqrt(((target - pred_tf) ** 2).mean())
    mae_tf = (target - pred_tf).abs().mean()
    mue_tf = mue(target, pred_tf)
    upf_tf = upf(target, pred_tf)

    # print
    print(
        f"XGBoost - RMSE: {rmse_xgb:.4f}, MAE: {mae_xgb:.4f}, MUPE: {mue_xgb:.4f}, UPF: {upf_xgb:.4f}"
    )
    print(
        f"Transformer - RMSE: {rmse_tf:.4f}, MAE: {mae_tf:.4f}, MUPE: {mue_tf:.4f}, UPF: {upf_tf:.4f}"
    )


if __name__ == "__main__":
    main()
