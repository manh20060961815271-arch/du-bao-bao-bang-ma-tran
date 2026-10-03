# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Typhoon track forecasting with a **Matrix Neural Network (MNN)**: predicts the storm-center position **12h ahead** (4 steps × 3h) from IBTrACS best-track data (`ibtracs-tbd.csv`, basins WP/EP/SP, 1940–2026, columns `ID, SEASON, ..., ISO_TIME, LAT, LON, STORM_SPEED, STORM_DIR`). Code comments, console output and plot labels are in Vietnamese.

## Running

There is no build system, test suite, linter, or requirements file. The whole pipeline (load → train → test → evaluate → plot) runs from one script:

```
python MNN_Main.py
```

Dependencies: `tensorflow` (used via `tensorflow.compat.v1` in graph mode), `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `cartopy`.

Outputs written to the repo root: `Training_Loss_Curve.png`, `Error_Distribution_Chart.png`, `R2_Scatter_Plot.png`, and one map per test storm in `Cyclone_Maps_All/`. The model checkpoint goes to the hard-coded Windows path `C:\tmp_cyclone_model`, which `Training.py` deletes (`shutil.rmtree`) and recreates on every run; `Testing.py` restores from the same path. `Y_pred_trueMNN.npy` / `Y_test.npy` are stale artifacts not produced by the current code.

No random seeds are set, so results vary between runs.

## Architecture

Data flow across modules:

1. **`MNN_Main.load_csv_data`** groups rows by `ID` into per-storm arrays of shape `(T, 5)`: `[Δlon, Δlat, speed, sin(dir), cos(dir)]`, where Δ is relative to the storm's **first** point (longitude wrapped to ±180°). Storms with < 8 points are dropped. Missing speed/dir are forward-filled per storm, then filled with 0. The start `(lon, lat)` of each storm is kept separately to reconstruct absolute positions later.
2. Split is by storm, in file order (≈ chronological): 70% train / 10% val / 20% test.
3. **`Data_Processing.data_processing`** builds sliding windows: window of `window_size=4` rows ending at anchor index `j`; lon/lat in the window are re-centered on the anchor; target is the **displacement** `pos[j+4] − pos[j]`. Shapes: `X (N, 4, 5)`, `Y (N, 1, 2)`.
4. **`Training.training`** fits `StandardScaler`s on train X (flattened to 5 features) and Y (2 features), builds the graph, trains, and returns the scalers + loss histories. Each storm is one mini-batch (variable batch size). Best val-loss checkpoint is saved.
5. **`Testing.testing`** rebuilds the **identical graph** and restores the checkpoint, returning per-storm true/pred displacements and `valid_indices` (indices into the test list of storms that produced ≥1 window).
6. Back in `MNN_Main`, displacements are added to the absolute anchor (`start + storm_data[j]`) — the loop `range(WINDOW_SIZE - 1, row_num - horizon)` must stay in sync with the loop in `data_processing`. Metrics: Haversine MAE/RMSE in km, per-storm error histogram, R² on absolute lat/lon, cartopy track maps.

### MNN layer (`Add_Layer.add_layer`)

Each output unit `(j,k)` is a bilinear form with a rank-1 weight: `u[b,j,k] = Σ_p Σ_q X[b,p,q]·W_L[j,k,p]·W_R[j,k,q] + θ[j,k]` (single `tf.einsum`). Optional LayerNorm over the `(m1, m2)` output matrix. Init std `(n1·n2)^-0.25`.

Shape configs are 3-element lists `[batch, rows, cols]`; the first element (`2` in `first_hid_cfg`/`second_hid_cfg`) is a dummy batch dim and is ignored. Network: `4×5 → 25×8 → 50×10 → 1×2`, activation `x/√(1+x²)` on hidden layers, linear output.

### Training scheme

"Two-stage" alternating optimization: two separate Adam optimizers (lr 1e-4, grads clipped to [-1, 1]). `train_op_L` updates every trainable variable whose name does **not** contain `W_R` (i.e. `W_L`, `theta`, `gamma`, `beta`); `train_op_R` updates only `W_R`. They run sequentially per batch. Variable names (`W_L`, `W_R`) are therefore load-bearing.

### Things to keep consistent

- The graph definition (layer configs, `layer_name`s, `use_layernorm`, activation) is duplicated in `Training.py` and `Testing.py`; changes must be mirrored in both or checkpoint restore fails.
- The feature count `5` is hard-coded in placeholders and scaler reshapes in both files, and `forecast_steps=4` / `horizon = 4` / `FORECAST_HOURS = 12` are set independently in `Data_Processing.py` and `MNN_Main.py`.
- The code assumes rows are evenly spaced at 3h.
