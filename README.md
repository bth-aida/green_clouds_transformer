# Green Clouds Transformer

A transformer-based implementation for predicting cpu usage in virtual machines for cloud environments. It is a PyTorch Transformer predicting multi-step model that incorporates a custom loss function (IdleAwareAsymmetricMSELoss) that penalize underprediction and weigh active periods higher than idle ones. Includes cross-validation grouped by instance ID to evaluate multi-step forecast performance (12 steps (1 hour)).

This repository also includes an XGBoost model for comparison.

## Getting Started

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
1. To setup a virtual environment:

   ```bash
   uv sync
   ```

1. Add data to a directory called `data`.

- To run the transformer model:

   ```bash
   uv run transformer.py
   ```

- To run the transformer model:

   ```bash
   uv run xgb_model.py
   ```

## Acknowledgements

This work is funded in part by the project “Green Clouds - Load Prediction and Optimization in Private Cloud Systems”, funded by the [Knowledge Foundation](https://www.kks.se) (grant: 20220215) in Sweden.
