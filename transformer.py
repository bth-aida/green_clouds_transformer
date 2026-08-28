import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data_processing import load_data


class IdleAwareAsymmetricMSELoss(nn.Module):
    def __init__(
        self,
        idle_threshold=0.02,
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
    def __init__(
        self,
        n_dynamic_features=2,
        n_static_features=2,
        d_model=64,
        n_heads=2,
        n_layers=2,
        horizon=12,
        dropout=0.1,
    ):
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
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
            nn.Sigmoid(),
        )
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
        elif torch.cuda.is_available():
            print("Using GPU")
            self.device = torch.device("cuda")
        else:
            print("Using CPU")
            self.device = torch.device("cpu")

    def set_model(self, model):
        self.model = model.to(self.device)

    def train_model(
        self,
        x_dynamic,
        x_static,
        y_train,
        val_data=None,
        epochs=20,
        batch_size=32,
        patience=5,
        min_delta=1e-4,
    ):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        criterion = nn.MSELoss()
        # criterion = nn.SmoothL1Loss()
        # criterion = nn.L1Loss()
        criterion = IdleAwareAsymmetricMSELoss(
            idle_threshold=0.02,
            idle_weight=0.2,
            active_weight=1.0,
            underprediction_weight=2.0,
        )

        # validation set
        val_loader = None
        if val_data is not None:
            x_val_d, x_val_s, y_val = val_data
            val_dataset = TensorDataset(x_val_d, x_val_s, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float("inf")
        best_model_weights = None
        patience_counter = 0

        dataset = TensorDataset(x_dynamic, x_static, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0

            for batch_dynamic, batch_static, batch_y in tqdm(
                loader, desc=f"Epoch {epoch}", leave=False
            ):
                batch_dynamic = batch_dynamic.to(self.device)
                batch_static = batch_static.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                preds = self.model(batch_dynamic, batch_static)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            # print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(loader):.6f}")
            train_loss /= len(loader)

            # validation (early stopping check)
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_dynamic, batch_static, batch_y in val_loader:
                        batch_dynamic = batch_dynamic.to(self.device)
                        batch_static = batch_static.to(self.device)
                        batch_y = batch_y.to(self.device)

                        preds = self.model(batch_dynamic, batch_static)
                        loss = criterion(preds, batch_y)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                print(
                    f"\nEpoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

                # improvement?
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    best_model_weights = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"No improvement ({patience_counter}/{patience})")

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    self.model.load_state_dict(best_model_weights)
                    break
            else:
                print(f"\nEpoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.6f}")

    @torch.no_grad()
    def test_model(
        self, x_test_dynamic, x_test_static, y_test, metadata, batch_size=32
    ):
        # run batches and predictions
        self.model.eval()
        dataset = TensorDataset(x_test_dynamic, x_test_static, y_test)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        preds_list, targets_list = [], []
        for batch_dynamic, batch_static, batch_y in loader:
            batch_dynamic = batch_dynamic.to(self.device)
            batch_static = batch_static.to(self.device)

            batch_y = batch_y.to(self.device)
            preds = self.model(batch_dynamic, batch_static)

            preds_list.append(preds.cpu())
            targets_list.append(batch_y.cpu())

        predictions = torch.cat(preds_list, dim=0).numpy()
        targets = torch.cat(targets_list, dim=0).numpy()

        # global error
        errors = predictions - targets
        avg_mse = float(np.mean(errors**2))
        avg_mae = float(np.mean(np.abs(errors)))

        # step 12 error
        step_12_errors = errors[:, 11]
        step_12_mse = float(np.mean(step_12_errors**2))
        step_12_mae = float(np.mean(np.abs(step_12_errors)))

        print(
            f"Test MSE: {avg_mse:.6f}, Test MAE: {avg_mae:.6f}, Step 12 MAE: {step_12_mae:.6f}"
        )

        results = {
            "mse": avg_mse,
            "mae": avg_mae,
            "step_12_mse": step_12_mse,
            "step_12_mae": step_12_mae,
            "predictions": predictions,
            "targets": targets,
        }

        # metadata
        csv_rows = []
        global_idx = 0

        for meta in metadata:
            series_id = meta["id_part"]
            start_ts = pd.to_datetime(meta["forecast_start"])

            for w in range(meta["n_windows"]):
                window_ts = start_ts + pd.Timedelta(minutes=5 * w)

                row = {
                    "idx": series_id,
                    "timestamp": window_ts,
                    "target": float(targets[global_idx, 0]),
                }

                for step_idx in range(12):
                    row[f"pred_{step_idx}"] = float(predictions[global_idx, step_idx])

                csv_rows.append(row)
                global_idx += 1

        df_out = pd.DataFrame(csv_rows)
        num_cols = [c for c in df_out.columns if c.startswith("pred_") or c == "target"]
        df_out[num_cols] = df_out[num_cols].round(3)
        df_out.to_csv("pred_tf.csv")

        return results

    def cross_validate(
        self,
        df: pd.DataFrame,
        n_splits=10,
        seq_len=48,
        horizon=12,
        epochs=20,
        batch_size=32,
        plot_each_fold=True,
    ):
        """
        Perform GroupKFold cross-validation by `id`.

        This avoids leakage from overlapping time windows by ensuring all rows
        for the same instance id stay in the same fold.
        """
        # dynamic_cols = ["cpu_usage", "mem_usage"]
        static_cols = ["mem_alloc", "vcpus"]

        df = df.sort_values(["id_part", "timestamp"]).reset_index(drop=True)

        gkf = GroupKFold(n_splits=n_splits)
        groups = df["id_part"]
        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            tqdm(gkf.split(df, groups=groups), total=n_splits, desc="Cross-validation"),
            start=1,
        ):
            print(f"\n--- Fold {fold_idx}/{n_splits} ---")

            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()

            scaler = StandardScaler()
            train_df[static_cols] = scaler.fit_transform(train_df[static_cols])
            val_df[static_cols] = scaler.transform(val_df[static_cols])

            x_dynamic, x_static, y_train, _ = self.prepare_dataset(
                train_df, seq_len=seq_len, horizon=horizon
            )
            x_val_d, x_val_s, y_val, val_meta = self.prepare_dataset(
                val_df, seq_len=seq_len, horizon=horizon
            )

            print(
                fold_idx,
                y_train.abs().mean().item(),
                y_val.abs().mean().item(),
            )

            if len(x_dynamic) == 0 or len(x_val_d) == 0:
                print("Skipping fold because it does not contain enough sequence data.")
                continue

            model = ForecastingTransformer(
                n_dynamic_features=x_dynamic.shape[-1],
                n_static_features=x_static.shape[-1],
                horizon=horizon,
            )
            self.set_model(model)

            self.train_model(
                x_dynamic,
                x_static,
                y_train,
                val_data=(x_val_d, x_val_s, y_val),
                epochs=epochs,
                batch_size=batch_size,
            )
            metrics = self.test_model(
                x_val_d, x_val_s, y_val, val_meta, batch_size=batch_size
            )

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

        if not fold_results:
            raise ValueError(
                "No valid folds were produced. Check dataset size, seq_len, and horizon."
            )

        mean_mse = float(np.mean([result["mse"] for result in fold_results]))
        std_mse = float(np.std([result["mse"] for result in fold_results]))
        mean_mae = float(np.mean([result["mae"] for result in fold_results]))
        std_mae = float(np.std([result["mae"] for result in fold_results]))
        mean_step_12_mse = float(
            np.mean([result["step_12_mse"] for result in fold_results])
        )
        std_step_12_mse = float(
            np.std([result["step_12_mse"] for result in fold_results])
        )
        mean_step_12_mae = float(
            np.mean([result["step_12_mae"] for result in fold_results])
        )
        std_step_12_mae = float(
            np.std([result["step_12_mae"] for result in fold_results])
        )

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

    def prepare_dataset(self, df: pd.DataFrame, seq_len: int, horizon: int):
        """Transforms a grouped and sorted DataFrame into tensors for Transformer training"""

        df["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24.0)

        df["minute_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.minute / 60.0)
        df["minute_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.minute / 60.0)

        df["day_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.dayofweek / 7.0)
        df["day_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.dayofweek / 7.0)

        dynamic_cols = [
            "cpu_usage",
            "mem_usage_ratio",
            "hour_sin",
            "hour_cos",
            "minute_sin",
            "minute_cos",
            "day_sin",
            "day_cos",
        ]
        # dynamic_cols = ["cpu_usage", "mem_usage_ratio"]
        static_cols = ["mem_alloc", "vcpus"]
        window_total = seq_len + horizon

        x_dynamics = []
        x_statics = []
        y_list = []
        meta_list = []

        # We use groupby to handle the windows within each unique instance
        for series_id, group in df.groupby("id_part", sort=False):
            group = group.sort_values("timestamp")
            static_data = group[static_cols].iloc[0].values

            # If the specific instance doesn't have enough data points, skip it
            if len(group) < window_total:
                continue

            # convert to tensors
            dynamic_tensor = torch.tensor(
                group[dynamic_cols].values, dtype=torch.float32
            )
            static_data = torch.tensor(
                group[static_cols].iloc[0].values, dtype=torch.float32
            )

            # create windows
            windows = dynamic_tensor.unfold(dimension=0, size=window_total, step=1)
            windows = windows.permute(0, 2, 1)

            x_windows = windows[:, :seq_len, :]
            y_windows = windows[:, seq_len:, 0]

            x_dynamics.append(x_windows)
            y_list.append(y_windows)

            # static data
            n_windows = x_windows.size(0)
            x_statics.append(static_data.repeat(n_windows, 1))

            # metadata
            meta_list.append(
                {
                    "id_part": series_id,
                    "forecast_start": group["timestamp"].iloc[seq_len],
                    "n_windows": n_windows,
                }
            )

        return (
            torch.cat(x_dynamics, dim=0),
            torch.cat(x_statics, dim=0),
            torch.cat(y_list, dim=0),
            meta_list,
        )


def cross_val():
    _, df_vm = load_data(start_index=0, size=10, clean=True)
    tf = TransformerWrapper()

    # kfold
    cv_results = tf.cross_validate(
        df=df_vm,
        n_splits=5,
        seq_len=96,
        horizon=12,
        epochs=20,
        batch_size=32,
    )

    print("\nFinal CV results:")
    print(cv_results)


def eval_tf():
    _, df_train = load_data(start_index=0, size=10, clean=True)
    _, df_test = load_data(start_index=10, size=5, clean=True)

    seq_len = 36
    horizon = 12

    tf = TransformerWrapper()
    x_train_d, x_train_s, y_train, _ = tf.prepare_dataset(
        df_train, seq_len=seq_len, horizon=horizon
    )
    x_test_d, x_test_s, y_test, test_meta = tf.prepare_dataset(
        df_test, seq_len=seq_len, horizon=horizon
    )

    # train model
    model = ForecastingTransformer(
        n_dynamic_features=x_train_d.shape[-1],
        n_static_features=x_train_s.shape[-1],
        horizon=horizon,
    )
    tf.set_model(model)

    print("--- training ---")
    tf.train_model(x_train_d, x_train_s, y_train, epochs=7, batch_size=32)

    # eval on testset
    print("\n--- test eval ---")
    metrics = tf.test_model(
        x_test_d, x_test_s, y_test, metadata=test_meta, batch_size=32
    )

    # metrics
    print("\n=== Result ===")
    print(f"Test MAE:  {metrics['step_12_mae']:.6f}")
    print(f"Test MSE:  {metrics['step_12_mse']:.6f}")
    print(f"Test RMSE: {np.sqrt(metrics['step_12_mse']):.6f}")


def main():
    # eval_tf()
    cross_val()


if __name__ == "__main__":
    main()
