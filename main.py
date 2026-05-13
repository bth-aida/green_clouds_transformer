import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from helper import group_df, load_cleura_db_files


class IdleAwareAsymmetricMSELoss(nn.Module):
    def __init__(
        self,
        idle_threshold=0.05,
        idle_weight=0.25,
        active_weight=1.0,
        underprediction_weight=2.0,
    ):
        super().__init__()
        self.idle_threshold = idle_threshold
        self.idle_weight = idle_weight
        self.active_weight = active_weight
        self.underprediction_weight = underprediction_weight

    def forward(self, preds, targets):
        error = preds - targets

        target_weights = torch.full_like(targets, self.idle_weight)
        target_weights[targets >= self.idle_threshold] = self.active_weight

        under_weights = torch.ones_like(error)
        under_weights[error < 0] = self.underprediction_weight
        return (target_weights * under_weights * error.pow(2)).mean()


class AsymmetricMSELoss(nn.Module):
    def __init__(self, underprediction_weight=2.0):
        super().__init__()
        self.underprediction_weight = underprediction_weight

    def forward(self, preds, targets):
        error = preds - targets
        weights = torch.ones_like(error)
        weights[error < 0] = self.underprediction_weight
        return (weights * error.pow(2)).mean()


class ForecastingTransformer(nn.Module):
    def __init__(self, n_dynamic_features=2, n_static_features=2, d_model=64, n_heads=2, n_layers=2, horizon=12, dropout=0.1):
        super().__init__()

        # Feature Embedding
        self.dynamic_embedding = nn.Linear(n_dynamic_features, d_model)
        self.static_embedding = nn.Sequential(
            nn.Linear(n_static_features, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Learnable 'Summary' Token (useful for variable length)
        self.summary_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional Encoding (Max length expected)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model))

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)

        # Direct Forecast Head
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, horizon), nn.Sigmoid())
        # self.head = nn.Linear(d_model, horizon)

    def forward(self, x_dynamic, x_static):
        batch_size = x_dynamic.size(0)
        seq_len = x_dynamic.size(1)

        # Embed features
        dynamic_repr = self.dynamic_embedding(x_dynamic)
        static_repr = self.static_embedding(x_static).unsqueeze(1)

        dynamic_repr = dynamic_repr + static_repr

        # Prepend the summary token to the sequence
        # This token will "collect" information from the whole sequence via attention
        s_token = self.summary_token.expand(batch_size, -1, -1)
        x = torch.cat((s_token, dynamic_repr), dim=1)

        # Add Positional Encoding (to the data part only, usually)
        x[:, 1:, :] += self.pos_encoding[:, :seq_len, :]

        # Pass through Transformer
        # If input is 1 step, x is now length 2 (summary + 1 data point)
        # If input is 60 steps, x is now length 61 (summary + 60 data points)
        feat = self.transformer(x)

        # Extract the summary token's final state for the forecast
        # summary_representation = feat[:, 0, :]
        # Do last token representation instead.
        sequence_representation = feat[:, -1, :]  # .mean(dim=1)

        return self.head(sequence_representation)


class TransformerWrapper:
    def __init__(self, seed=42):
        self.seed = seed
        self.rng = random.Random(seed)
        np.random.seed(seed)

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def set_model(self, model):
        self.model = model.to(self.device)

    def train_model(self, x_dynamic, x_static, y_train, epochs=20, batch_size=32):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        criterion = nn.MSELoss()
        # criterion = nn.SmoothL1Loss()
        # criterion = nn.L1Loss()
        criterion = IdleAwareAsymmetricMSELoss(
            idle_threshold=0.05,
            idle_weight=0.2,
            active_weight=1.0,
            underprediction_weight=2.0,
        )

        dataset = TensorDataset(x_dynamic, x_static, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_dynamic, batch_static, batch_y in loader:
                batch_dynamic = batch_dynamic.to(self.device)
                batch_static = batch_static.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                preds = self.model(batch_dynamic, batch_static)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.6f}")

    @torch.no_grad()
    def test_model(self, x_test_dynamic, x_test_static, y_test, metadata=None, batch_size=32):
        criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()

        dataset = TensorDataset(x_test_dynamic, x_test_static, y_test)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        total_mse = 0.0
        total_mae = 0.0
        all_preds = []
        all_targets = []

        for batch_dynamic, batch_static, batch_y in loader:
            batch_dynamic = batch_dynamic.to(self.device)
            batch_static = batch_static.to(self.device)
            batch_y = batch_y.to(self.device)

            preds = self.model(batch_dynamic, batch_static)

            total_mse += criterion(preds, batch_y).item()
            total_mae += mae_criterion(preds, batch_y).item()

            all_preds.append(preds.cpu())
            all_targets.append(batch_y.cpu())

        avg_mse = total_mse / len(loader)
        avg_mae = total_mae / len(loader)

        predictions = torch.cat(all_preds, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()

        # for p, t in zip(predictions, targets):
        #    print(p[-1], t[-1])

        per_step_mse = np.mean((predictions - targets) ** 2, axis=0)
        per_step_mae = np.mean(np.abs(predictions - targets), axis=0)

        step_12_index = 11
        step_12_mse = float(per_step_mse[step_12_index])
        step_12_mae = float(per_step_mae[step_12_index])

        results = {
            "mse": float(avg_mse),
            "mae": float(avg_mae),
            "per_step_mse": per_step_mse.tolist(),
            "per_step_mae": per_step_mae.tolist(),
            "step_12_mse": step_12_mse,
            "step_12_mae": step_12_mae,
            "predictions": predictions,
            "targets": targets,
        }

        if metadata is not None:
            forecast_rows = []
            per_series = {}

            for i, meta in enumerate(metadata):
                series_id = meta["id"]
                timestamps = meta["target_timestamps"]

                if series_id not in per_series:
                    per_series[series_id] = {
                        "preds": [],
                        "targets": [],
                        "step_12_preds": [],
                        "step_12_targets": [],
                    }

                per_series[series_id]["preds"].append(predictions[i])
                per_series[series_id]["targets"].append(targets[i])
                per_series[series_id]["step_12_preds"].append(predictions[i, step_12_index])
                per_series[series_id]["step_12_targets"].append(targets[i, step_12_index])

                for step_idx in range(predictions.shape[1]):
                    forecast_rows.append(
                        {
                            "id": series_id,
                            "forecast_start": meta["forecast_start"],
                            "horizon_step": step_idx + 1,
                            "target_timestamp": timestamps[step_idx],
                            "prediction": float(predictions[i, step_idx]),
                            "target": float(targets[i, step_idx]),
                            "abs_error": float(abs(predictions[i, step_idx] - targets[i, step_idx])),
                            "sq_error": float((predictions[i, step_idx] - targets[i, step_idx]) ** 2),
                        }
                    )

            per_series_metrics = []
            for series_id, values in per_series.items():
                preds_arr = np.array(values["preds"])
                targets_arr = np.array(values["targets"])
                step_12_preds = np.array(values["step_12_preds"])
                step_12_targets = np.array(values["step_12_targets"])

                per_series_metrics.append(
                    {
                        "id": series_id,
                        "n_windows": len(preds_arr),
                        "mse": float(np.mean((preds_arr - targets_arr) ** 2)),
                        "mae": float(np.mean(np.abs(preds_arr - targets_arr))),
                        "step_12_mse": float(np.mean((step_12_preds - step_12_targets) ** 2)),
                        "step_12_mae": float(np.mean(np.abs(step_12_preds - step_12_targets))),
                    }
                )

            results["per_series_metrics"] = per_series_metrics
            results["forecast_rows"] = forecast_rows

        print(f"Test MSE: {avg_mse:.6f}, Test MAE: {avg_mae:.6f}, Step 12 MAE: {step_12_mae:.6f}")
        return results

    def cross_validate(self, df, n_splits=10, seq_len=48, horizon=12, epochs=20, batch_size=32, plot_each_fold=True):
        """
        Perform GroupKFold cross-validation by `id`.

        This avoids leakage from overlapping time windows by ensuring all rows
        for the same instance id stay in the same fold.
        """
        dynamic_cols = ["cpu_usage", "mem_usage"]
        static_cols = ["mem_alloc", "vcpus"]

        df = df.copy()

        if "timestamp" in df.columns:
            df = df.sort_values(["id", "timestamp"]).reset_index(drop=True)
        else:
            df = df.sort_values(["id"]).reset_index(drop=True)

        gkf = GroupKFold(n_splits=n_splits)
        groups = df["id"]

        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups), start=1):
            print(f"\n--- Fold {fold_idx}/{n_splits} ---")

            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()
            train_df["mem_usage"] = np.where(train_df["mem_alloc"] > 0, train_df["mem_usage"] / train_df["mem_alloc"], 0.0)
            train_df["cpu_usage"] = train_df["cpu_usage"] / 100

            val_df["mem_usage"] = np.where(val_df["mem_alloc"] > 0, val_df["mem_usage"] / val_df["mem_alloc"], 0.0)
            val_df["cpu_usage"] = val_df["cpu_usage"] / 100

            scaler = StandardScaler()
            train_df[static_cols] = scaler.fit_transform(train_df[static_cols])
            val_df[static_cols] = scaler.transform(val_df[static_cols])

            x_dynamic, x_static, y_train, _ = self.prepare_dataset(train_df, seq_len=seq_len, horizon=horizon)
            x_val_d, x_val_s, y_val, val_meta = self.prepare_dataset(val_df, seq_len=seq_len, horizon=horizon)

            print(
                fold_idx,
                y_train.abs().mean().item(),
                y_val.abs().mean().item(),
            )

            if len(x_dynamic) == 0 or len(x_val_d) == 0:
                print("Skipping fold because it does not contain enough sequence data.")
                continue

            model = ForecastingTransformer(n_dynamic_features=x_dynamic.shape[-1], n_static_features=x_static.shape[-1], horizon=horizon)
            self.set_model(model)

            self.train_model(x_dynamic, x_static, y_train, epochs=epochs, batch_size=batch_size)
            metrics = self.test_model(x_val_d, x_val_s, y_val, metadata=val_meta, batch_size=batch_size)

            fold_result = {
                "fold": fold_idx,
                "mse": metrics["mse"],
                "mae": metrics["mae"],
                "step_12_mse": metrics["step_12_mse"],
                "step_12_mae": metrics["step_12_mae"],
                "n_train_samples": len(x_dynamic),
                "n_val_samples": len(x_val_d),
            }
            fold_results.append(fold_result)

            print(
                f"Fold {fold_idx} complete - "
                f"MSE: {metrics['mse']:.6f}, "
                f"MAE: {metrics['mae']:.6f}, "
                f"Step12 MSE: {metrics['step_12_mse']:.6f}, "
                f"Step12 MAE: {metrics['step_12_mae']:.6f}, "
                f"train samples: {len(x_dynamic)}, "
                f"val samples: {len(x_val_d)}"
            )
            if plot_each_fold:
                self.plot_random_validation_series(
                    forecast_rows=metrics.get("forecast_rows", []),
                    fold_idx=fold_idx,
                    filename=True,
                    horizon_step=12,
                )

        if not fold_results:
            raise ValueError("No valid folds were produced. Check dataset size, seq_len, and horizon.")

        mean_mse = float(np.mean([result["mse"] for result in fold_results]))
        std_mse = float(np.std([result["mse"] for result in fold_results]))
        mean_mae = float(np.mean([result["mae"] for result in fold_results]))
        std_mae = float(np.std([result["mae"] for result in fold_results]))
        mean_step_12_mse = float(np.mean([result["step_12_mse"] for result in fold_results]))
        std_step_12_mse = float(np.std([result["step_12_mse"] for result in fold_results]))
        mean_step_12_mae = float(np.mean([result["step_12_mae"] for result in fold_results]))
        std_step_12_mae = float(np.std([result["step_12_mae"] for result in fold_results]))

        print("\n=== Cross-validation summary ===")
        for result in fold_results:
            print(
                f"Fold {result['fold']}: "
                f"MSE={result['mse']:.6f}, "
                f"MAE={result['mae']:.6f}, "
                f"Step12 MSE={result['step_12_mse']:.6f}, "
                f"Step12 MAE={result['step_12_mae']:.6f}, "
                f"train={result['n_train_samples']}, "
                f"val={result['n_val_samples']}"
            )

        print(f"Mean MSE: {mean_mse:.6f} ± {std_mse:.6f}")
        print(f"Mean MAE: {mean_mae:.6f} ± {std_mae:.6f}")
        print(f"Mean Step12 MSE: {mean_step_12_mse:.6f} ± {std_step_12_mse:.6f}")
        print(f"Mean Step12 MAE: {mean_step_12_mae:.6f} ± {std_step_12_mae:.6f}")

        return {
            "fold_results": fold_results,
            "mean_mse": mean_mse,
            "std_mse": std_mse,
            "mean_mae": mean_mae,
            "std_mae": std_mae,
            "mean_step_12_mse": mean_step_12_mse,
            "std_step_12_mse": std_step_12_mse,
            "mean_step_12_mae": mean_step_12_mae,
            "std_step_12_mae": std_step_12_mae,
        }

    def prepare_dataset(self, df, seq_len=48, horizon=12):
        """
        Transforms a grouped and sorted DataFrame into tensors for Transformer training.
        """
        # Features: order must be consistent for eval
        dynamic_cols = ["cpu_usage", "mem_usage"]
        static_cols = ["mem_alloc", "vcpus"]

        all_x_dynamic = []
        all_x_static = []
        all_y = []
        all_meta = []

        # We use groupby to handle the windows within each unique instance
        for series_id, group in df.groupby("id", sort=False):
            if "timestamp" in group.columns:
                group = group.sort_values("timestamp")
            else:
                group = group.reset_index(drop=True)

            dynamic_data = group[dynamic_cols].values
            static_data = group[static_cols].iloc[0].values

            # If the specific instance doesn't have enough data points, skip it
            if len(group) < (seq_len + horizon):
                continue

            # Efficient sliding window using numpy indexing
            for i in range(len(group) - seq_len - horizon + 1):
                # Input features (index i to i + seq_len)
                x_window = dynamic_data[i : i + seq_len]

                # Target CPU usage (next 12 steps starting right after x_window)
                # Index 0 is 'cpu_usage'
                y_window = dynamic_data[i + seq_len : i + seq_len + horizon, 0]

                target_rows = group.iloc[i + seq_len : i + seq_len + horizon]
                target_timestamps = (
                    target_rows["timestamp"].tolist() if "timestamp" in group.columns else list(range(i + seq_len, i + seq_len + horizon))
                )

                all_x_dynamic.append(x_window)
                all_x_static.append(static_data)
                all_y.append(y_window)
                all_meta.append(
                    {
                        "id": series_id,
                        "forecast_start": target_timestamps[0],
                        "target_timestamps": target_timestamps,
                    }
                )

        # Stack into final tensors
        if not all_x_dynamic:
            return torch.empty(0), torch.empty(0), torch.empty(0), []

        return (
            torch.tensor(np.array(all_x_dynamic), dtype=torch.float32),
            torch.tensor(np.array(all_x_static), dtype=torch.float32),
            torch.tensor(np.array(all_y), dtype=torch.float32),
            all_meta,
        )

    def plot_random_validation_series(self, forecast_rows, fold_idx, filename=False, horizon_step=12):
        if not forecast_rows:
            print(f"No forecast rows available to plot for fold {fold_idx}.")
            return

        rows = [row for row in forecast_rows if row["horizon_step"] == horizon_step]
        if not rows:
            print(f"No forecast rows found for horizon step {horizon_step} in fold {fold_idx}.")
            return

        series_ids = sorted({row["id"] for row in rows})
        if not series_ids:
            print(f"No series ids available to plot for fold {fold_idx}.")
            return
        if filename:
            ids = series_ids
        else:
            ids = [self.rng.choice(series_ids)]
        for chosen_id in ids:
            series_rows = [row for row in rows if row["id"] == chosen_id]
            series_rows = sorted(series_rows, key=lambda row: row["target_timestamp"])

            x = [row["target_timestamp"] for row in series_rows]
            y_true = np.array([row["target"] for row in series_rows], dtype=float)
            y_pred = np.array([row["prediction"] for row in series_rows], dtype=float)

            errors = y_pred - y_true
            mae = np.mean(np.abs(errors))
            mse = np.mean(np.square(errors))
            rmse = np.sqrt(mse)
            metrics_text = f"MAE: {mae:.4f}    MSE: {mse:.4f}    RMSE: {rmse:.4f}"

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(x, y_true, label="Actual", linewidth=2)
            ax.plot(x, y_pred, label=f"Predicted (step {horizon_step})", linewidth=2, alpha=0.8)

            ax.set_title(f"Fold {fold_idx} - Random validation series: {chosen_id} - Step {horizon_step}")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("CPU Usage")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            fig.subplots_adjust(bottom=0.22)
            fig.text(0.5, 0.06, metrics_text, ha="center", va="center")

            if filename:
                plt.savefig(f"timeseries/{chosen_id}.png", bbox_inches="tight")
            else:
                plt.show()
            plt.close()


if __name__ == "__main__":
    _, instances_df = load_cleura_db_files(limit=4)
    tf = TransformerWrapper()

    if instances_df.empty:
        raise ValueError("No instance data was loaded from helper.load_cleura_db_files().")

    df = group_df(instances_df)

    tf = TransformerWrapper()

    cv_results = tf.cross_validate(
        df=df,
        n_splits=5,
        seq_len=96,
        horizon=12,
        epochs=10,
        batch_size=32,
    )

    print("\nFinal CV results:")
    print(cv_results)
