# external imports
from glob import glob

import pandas as pd
import pyarrow.json as paj
from tqdm import tqdm


def load_data(
    dir_path: str = "../cleura",
    start_index: int = 0,
    size: int | None = None,
    clean: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the contents of all .db files (which are JSONL) in the specified directory.
    Returns two DataFrames: one for the hypervisor data and one for the flattened vm data.

    Parameters
    ----------
    dir_path : str
        Folder containing the JSON files.
    start_index : int
        Index of the first file to read (after sorting).
    size : int | None
        Number of files to load from `start_index`.  ``None`` loads all remaining files.
    -------
    Returns:
        tuple: (df_hv, df_vm)
               df_hv contains hypervisor data.
               df_vm contains flattened vm data with a timestamp column.
    """
    # load file names for .db files
    db_files = glob(dir_path + "/*.db")
    db_files.sort()
    db_files = db_files[start_index:]

    if size is not None:
        db_files = db_files[:size]

    # lists for dfs
    hv_df_list: list[pd.DataFrame] = []
    vm_df_list: list[pd.DataFrame] = []

    # read each file
    for db_file in tqdm(db_files, desc="Loading db-files"):
        # read using pyarrow
        table = paj.read_json(db_file)  # type: ignore
        df_file = table.to_pandas()

        # convert timestamp column to datetime object, sort
        df_file["timestamp"] = pd.to_datetime(df_file["timestamp"]).dt.tz_localize(None)
        df_file.sort_values(["timestamp", "identifier"], inplace=True)

        # save to list
        hv_df_list.append(
            df_file[
                [
                    "timestamp",
                    "identifier",
                    "cpu_usage",
                    "memory_total",
                    "memory_usage",
                    "num_instances",
                ]
            ]
        )

        # explode vm instances, save to list
        df_temp = df_file[["timestamp", "instances", "identifier"]].explode("instances")
        df_vm = pd.DataFrame(df_temp["instances"].tolist(), index=df_temp.index)
        df_vm = pd.concat([df_temp.drop(columns=["instances"]), df_vm], axis=1)
        df_vm.drop(columns="name", inplace=True)
        df_vm = df_vm.reset_index(drop=True)
        vm_df_list.append(df_vm)

    # hypervisor df
    df_hv = pd.concat(hv_df_list, ignore_index=True).reset_index(drop=True)

    # create instances dataframe
    df_vm = pd.concat(vm_df_list, ignore_index=True).reset_index(drop=True)

    if clean:
        df_vm = clean_data(df_vm)

    return df_hv, df_vm


def clean_data(df_vm: pd.DataFrame, min_count: int = 25):
    """Remove negative cpu values, normalize cpu values,
    factorize hypervisor, add features, round time"""
    df = df_vm.copy()

    # remove negative cpu usage values
    df["cpu_usage"] = df["cpu_usage"].clip(lower=0)

    # set cpu usage to 0-1 (most of the time)
    df["cpu_usage"] = df["cpu_usage"] / 100

    # set hypervisor to 0 or 1
    df["identifier"] = pd.factorize(df["identifier"])[0]

    # new feature
    df["mem_usage_ratio"] = df["mem_usage"] / df["mem_alloc"]
    df.drop(columns=["mem_usage"], inplace=True)

    df_list = []
    for key, item in df.groupby("id"):
        df = item.reset_index(drop=True)

        # round to nearest 5 minutes
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.round("5min")

        # find gaps
        df["time_diff"] = df["timestamp"].diff().dt.total_seconds() / 60.0
        df["new_part"] = (df["time_diff"] > 5) | (df["time_diff"].isna())
        df["part_id"] = df["new_part"].cumsum()

        for i, part_df in df.groupby("part_id"):
            clean_df = part_df.drop(columns=["time_diff", "new_part", "part_id"])
            clean_df["id_part"] = clean_df["id"].astype(str) + "_" + str(i)
            clean_df.reset_index(drop=True, inplace=True)

            if len(clean_df) > min_count:
                df_list.append(clean_df)

    df = pd.concat(df_list)
    return df


def add_features_tree(
    df: pd.DataFrame,
    cpu_lags: int = 12,
    mem_lags: int = 3,
    rolling_count: int = 6,
):
    """Add features to db-dataframe for tree models"""
    df = df.copy()

    # time features
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    grouped_cpu = df.groupby("id_part")["cpu_usage"]
    grouped_mem = df.groupby("id_part")["mem_usage_ratio"]

    # lag features
    for lag in range(1, cpu_lags + 1):
        df[f"cpu_usage_lag{lag}"] = grouped_cpu.shift(lag)
    for lag in range(1, mem_lags + 1):
        df[f"mem_ratio_lag{lag}"] = grouped_mem.shift(lag)

    # rolling features
    df["cpu_usage_rolling_mean"] = grouped_cpu.transform(
        lambda x: x.shift(1).rolling(rolling_count).mean()
    )
    df["cpu_usage_rolling_std"] = grouped_cpu.transform(
        lambda x: x.shift(1).rolling(rolling_count).std()
    )
    df["mem_usage_ratio_rolling_mean"] = grouped_mem.transform(
        lambda x: x.shift(1).rolling(rolling_count).mean()
    )
    df["mem_usage_ratio_rolling_std"] = grouped_mem.transform(
        lambda x: x.shift(1).rolling(rolling_count).std()
    )

    float_cols = df.select_dtypes(include=["float"]).columns
    df[float_cols] = df[float_cols].round(6)

    # df.drop(columns=["mem_alloc"], inplace=True)

    # sort on time
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df
