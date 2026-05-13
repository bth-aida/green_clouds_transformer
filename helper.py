import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_cleura_db_files(directory="../cleura/", limit=None):
    """
    Loads the contents of all .db files (which are JSONL) in the specified directory.
    Returns two DataFrames: one for the summary data and one for the flattened instance data.

    Args:
        directory (str): The path to the directory containing .db files.

    Returns:
        tuple: (summary_df, instances_df)
               summary_df contains top-level metrics.
               instances_df contains flattened instance data with a timestamp column.
    """
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return pd.DataFrame(), pd.DataFrame()

    # Create the pattern for .db files
    pattern = os.path.join(directory, "*.db")
    db_files = glob.glob(pattern)

    if not db_files:
        print(f"No .db files found in {directory}")
        return pd.DataFrame(), pd.DataFrame()

    all_summaries = []
    all_instances = []
    c = 0
    for db_file in db_files:
        try:
            with open(db_file, "r") as f:
                # Read line by line to handle JSON Lines (JSONL) format
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"Skipping line in {os.path.basename(db_file)} due to JSON decode error: {e}")
                        continue

                    # 1. Handle Summary Data (Top-level)
                    # We extract everything except the 'instances' list
                    summary_entry = {k: v for k, v in entry.items() if k != "instances"}
                    summary_entry["source_file"] = os.path.basename(db_file)
                    all_summaries.append(summary_entry)

                    # 2. Handle Instances Data (Nested)
                    if "instances" in entry and isinstance(entry["instances"], list):
                        instances = entry["instances"]
                        # Create a DataFrame from instances
                        instances_df = pd.DataFrame(instances)

                        # Add the timestamp to every instance row to link it back to the summary
                        timestamp = entry.get("timestamp")
                        hypervisor = entry.get("identifier")
                        if timestamp:
                            instances_df["timestamp"] = timestamp

                        instances_df["source_file"] = os.path.basename(db_file)
                        instances_df["hypervisor"] = hypervisor
                        all_instances.append(instances_df)
            if c == limit:
                break
            else:
                c = c + 1
        except Exception as e:
            print(f"Error processing file {db_file}: {e}")

    summary_df = pd.DataFrame(all_summaries) if all_summaries else pd.DataFrame()
    instances_df = pd.concat(all_instances, ignore_index=True) if all_instances else pd.DataFrame()

    return summary_df, instances_df


def group_df(instances_df):
    instances_grouped_df = (
        instances_df.groupby(["id", "timestamp"])
        .agg(
            source_file=("source_file", "first"),
            hypervisor=("hypervisor", "first"),
            **{
                col: pd.NamedAgg(column=col, aggfunc="first")
                for col in instances_df.columns
                if col not in ["id", "timestamp", "source_file", "hypervisor"]
            },
        )
        .reset_index()
    )
    return instances_grouped_df


if __name__ == "__main__":
    summary_df, instances_df = load_cleura_db_files()

    print("--- Summary DataFrame ---")
    print(summary_df.head())

    print("\n--- Instances DataFrame ---")
    print(instances_df.head())
    print(instances_df.shape, len(set(instances_df.id)))
    print(instances_df[["mem_alloc", "mem_usage", "vcpus", "cpu_usage"]].describe())

    instances_grouped_df = group_df(instances_df)

    print(instances_grouped_df[instances_grouped_df.id == "000028e3"])

    plot_histogram = False
    plot_timeseries = True

    if plot_histogram:
        plt.figure(figsize=(12, 6))
        # Using seaborn countplot would be easier, but sticking to matplotlib for minimal change
        # We use histograms for distribution analysis.
        sns.kdeplot(data=instances_df[instances_df["vcpus"] == 1], x="cpu_usage", label="1 vCPU", fill=True, alpha=0.5)
        sns.kdeplot(data=instances_df[instances_df["vcpus"] == 2], x="cpu_usage", label="2 vCPUs", fill=True, alpha=0.5)
        sns.kdeplot(data=instances_df[instances_df["vcpus"] == 4], x="cpu_usage", label="4 vCPUs", fill=True, alpha=0.5)
        # For vCPUs >= 8, we select the maximum vCPU count present for labeling purposes
        max_vcpus = instances_df["vcpus"].max()
        if max_vcpus > 4:
            sns.kdeplot(data=instances_df[instances_df["vcpus"] == max_vcpus], x="cpu_usage", label=f"5-{max_vcpus} vCPUs", fill=True, alpha=0.5)

        plt.title("Distribution of CPU Usage Grouped by vCPUs")
        plt.xlabel("CPU Usage")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(axis="y", alpha=0.5)
        plt.show()
    else:
        print("Instances DataFrame is empty, skipping plotting.")

    if plot_timeseries:
        instances_grouped_df["time_step"] = instances_grouped_df.groupby("id").cumcount()
        mean_timeseries = instances_grouped_df.groupby(["vcpus", "time_step"])["cpu_usage"].mean().reset_index()

        plt.figure(figsize=(14, 7))

        # Pivot the dataframe to make 'vcpus' a column/grouping variable for seaborn
        sns.lineplot(data=mean_timeseries, x="time_step", y="cpu_usage", hue="vcpus", palette="tab10")

        plt.title("Mean CPU Usage Timeseries Grouped by vCPUs")
        plt.xlabel("time_step")
        plt.ylabel("Mean CPU Usage")
        plt.legend(title="vCPUs")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()
        plt.show()
    if plot_timeseries:
        df_test = instances_grouped_df[instances_grouped_df.vcpus == 2]
        df_test["time_step"] = df_test.groupby("id").cumcount()

        plt.figure(figsize=(14, 7))

        # Pivot the dataframe to make 'vcpus' a column/grouping variable for seaborn
        sns.lineplot(data=df_test, x="time_step", y="cpu_usage", hue="id", alpha=0.05)

        plt.title("Mean CPU Usage Timeseries Grouped by vCPUs")
        plt.xlabel("time_step")
        plt.ylabel("Mean CPU Usage")
        plt.legend("ID")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()
        plt.show()
