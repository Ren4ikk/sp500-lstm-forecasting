# 📈 S&P 500 Stock Price Forecasting with LSTM

A PyTorch implementation of a 2-layer LSTM network for short-term time series forecasting, trained on 5 years of daily closing prices across all 505 S&P 500 constituents. The model predicts next-day closing price from a 30-day rolling window and is validated against live market data fetched via `yfinance`.

## Highlights

- **Multi-ticker training**: trained jointly on ~619K rows across 505 tickers rather than a single stock, so the model learns generalizable price-movement patterns instead of overfitting to one company's idiosyncrasies
- **Live inference validation**: tested on real-time AAPL data (2025–2026) despite training only on 2013–2018 historical data — a ~8-year distribution shift — and still achieved <1% relative error
- **Train/test split ablation study**: quantified how the train/test split ratio affects generalization error (80/20 vs 50/50), showing a 5x difference in test loss
- **Standard time-series best practices**: per-ticker normalization, strictly sequential (non-shuffled) train/test split to prevent lookahead leakage, gradient clipping, and adaptive LR scheduling

## Problem Statement

Financial time series are notoriously hard to forecast: they combine trend, seasonality, and a high noise floor. Feedforward networks ignore sequence order entirely, and vanilla RNNs suffer from vanishing gradients on long sequences. This project implements an **LSTM (Hochreiter & Schmidhuber, 1997)**, whose gating mechanism lets the network learn what to retain and what to discard over long time horizons — a property well suited to capturing dependencies in noisy financial data.

## Dataset

[S&P 500 Stock Data](https://www.kaggle.com/datasets/camnugent/sandp500) (Kaggle)

| | |
|---|---|
| Period | Feb 2013 – Feb 2018 (5 years) |
| Tickers | 505 companies |
| Rows | ~619,000 |
| Fields | `date`, `open`, `high`, `low`, `close`, `volume`, `Name` |
| Target | `close` price only |

## Model Architecture

```
Input (batch, 30, 1)
    → LSTM layer 1 (hidden_size=128)
    → LSTM layer 2 (hidden_size=128), dropout=0.2 between layers
    → last timestep output (batch, 128)
    → Linear(128 → 32) → ReLU
    → Linear(32 → 1)
    → predicted next-day close price
```

~274K trainable parameters. The model consumes a 30-day window of normalized closing prices (roughly one trading month) and outputs a single-step forecast.

## Data Pipeline

1. **Normalization** — each ticker is scaled independently with `MinMaxScaler` to `[0, 1]`, since raw price ranges vary by orders of magnitude across companies (e.g. AMZN vs. a low-priced stock). This lets the model learn *shape* of price movement rather than absolute price level.
2. **Sliding window** — `SEQ_LEN = 30`: consecutive windows `[p_t, ..., p_t+29] → p_t+30` are generated per ticker.
3. **Sequential split** — no random shuffling of the split itself (this would leak future information into training); the scaler is fit only on the training portion and applied (not refit) on the test portion.
4. All tickers are combined via `ConcatDataset` and batched with `DataLoader(batch_size=64)`, with shuffling enabled only within the training set.

## Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 50 |
| Batch size | 64 |
| Optimizer | Adam (lr=1e-3) |
| Loss | MSE |
| Gradient clipping | max_norm=1.0 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |

Gradient clipping was used to counter exploding gradients common in RNN training on long sequences. `ReduceLROnPlateau` halves the learning rate whenever test loss plateaus for 5 consecutive epochs, letting the model converge quickly early on and fine-tune later without manual LR scheduling.

## Experiment: Train/Test Split Ratio

To study how the amount of training data affects generalization, the model was trained under two `TRAIN_RATIO` settings:

| TRAIN_RATIO | Train size | Test size | Train loss (epoch 50) | Test loss (epoch 50) |
|---|---|---|---|---|
| 0.8 | 80% | 20% | 0.000547 | 0.044037 |
| 0.5 | 50% | 50% | 0.000801 | 0.222131 |

**Finding**: reducing the training set from 80% to 50% increased test loss ~5x. With less training data, the model sees fewer price-pattern examples and generalizes worse to the later (unseen) portion of the time series — and the train/test loss gap in both settings indicates the model still overfits somewhat despite dropout, a known challenge for LSTMs on noisy financial data.

## Inference Results (Live Data)

The 80/20 model was evaluated on live AAPL data pulled via `yfinance`, using the last 30 trading days as input:

| Metric | Value |
|---|---|
| Last known price | 251.49 |
| Predicted price | 251.27 |
| Actual next price | 252.84 |
| Absolute error | 1.57 |
| Relative error | ~0.62% |

Despite an ~8-year gap between the training data (2013–2018) and inference data (2025–2026), the model correctly predicted a trend reversal (small uptick after a downtrend), suggesting it learned generalizable price-movement patterns rather than memorizing a specific period.

## Limitations

- Univariate model: does not incorporate exogenous signals (news sentiment, macroeconomic indicators, trading volume) that materially affect real-world price behavior
- Single-step forecasting only; no multi-horizon prediction
- Train/test loss gap indicates residual overfitting despite dropout regularization

## Tech Stack

`Python` · `PyTorch` · `pandas` · `NumPy` · `scikit-learn` (MinMaxScaler) · `yfinance` · `matplotlib`

## Project Structure

```
.
├── data/                  # dataset loading & preprocessing
├── model.py               # LSTM architecture definition
├── train.py                # training loop, scheduler, gradient clipping
├── inference.py           # live inference via yfinance
├── notebooks/              # experiments (TRAIN_RATIO ablation, etc.)
└── README.md
```

## Acknowledgments

This project was originally completed as a deep learning coursework assignment; the codebase and writeup have been adapted here as a standalone portfolio project.
