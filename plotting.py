import pandas as pd
import plotly.graph_objects as pgo

from data_processing import clean_data, load_data


def plot_results(df: pd.DataFrame, id: str):
    df = df[df["id"] == id]
    fig = pgo.Figure()
    cols = ["cpu_usage", "pred_1"]

    for col in cols:
        fig.add_trace(
            pgo.Scattergl(
                x=df["timestamp"],
                y=df[col],
                mode="lines+markers",
                name=col,
            )
        )

    fig.update_layout(
        title="CPU Prediction Results",
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode="x unified",
        dragmode="pan",
        xaxis={
            "fixedrange": False,
            "rangeslider": {"visible": True},
        },
        yaxis={"fixedrange": False},
    )

    fig.show(config={"scrollZoom": True})


def plot_vm_ids(df_vm: pd.DataFrame, vm_ids: list[str]):
    fig = pgo.Figure()

    # filter out ids
    df_filtered = df_vm[df_vm["id"].isin(vm_ids)]

    # group and create plots
    for vm_id, group in df_filtered.groupby("id"):
        fig.add_trace(
            pgo.Scattergl(
                x=group["timestamp"],
                y=group["cpu_usage"],
                mode="lines+markers",
                name=str(vm_id),
            )
        )

    fig.update_layout(
        title="Timeserii",
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode="x unified",
        dragmode="pan",
        xaxis={
            "fixedrange": False,
            "rangeslider": {"visible": True},
        },
        yaxis={"fixedrange": False},
    )

    fig.show(config={"scrollZoom": True})


def plot_vm():
    _, df_vm = load_data(size=20)
    df_vm_clean = clean_data(df_vm)

    ids = df_vm_clean["id"].unique().tolist()[:5]
    print(ids)
    plot_vm_ids(df_vm_clean, ids)


def plot_result():
    df = pd.read_csv("results_XGBoost.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    plot_results(df, "00304c03")


def main():
    plot_vm()


if __name__ == "__main__":
    main()
