import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from model import LSTMModel
from data import SEQ_LEN, TRAIN_RATIO, load_data

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "lstm_stock.pt"
CSV_PATH   = "all_stocks_5yr.csv"

MODE   = "yfinance"   # "yfinance" | "csv" | "manual"
TICKER = "AAPL"

MY_PRICES = [
]


def _load_model():
    model = LSTMModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def _infer(model, prices: list) -> float:
    arr    = np.array(prices).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(arr)
    x      = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_scaled = model(x).cpu().numpy()
    return float(scaler.inverse_transform(pred_scaled)[0, 0])


def _fetch_yfinance(ticker: str) -> tuple[list, float | None]:
    import yfinance as yf
    df = yf.download(ticker, period=f"{SEQ_LEN + 10}d", interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"yfinance returned no data for ticker '{ticker}'")
    closes = df["Close"].dropna().values.flatten()
    if len(closes) < SEQ_LEN + 1:
        return closes[-SEQ_LEN:].tolist(), None
    return closes[-(SEQ_LEN + 1):-1].tolist(), float(closes[-1])


def _fetch_csv(ticker: str) -> tuple[list, float | None]:
    raw   = load_data(CSV_PATH, ticker).flatten()
    split = int(len(raw) * TRAIN_RATIO)
    test  = raw[split:]
    assert len(test) > SEQ_LEN, "Not enough test data for this ticker"
    return test[-(SEQ_LEN + 1):-1].tolist(), float(test[-1])


def run():
    model = _load_model()

    if MODE == "yfinance":
        print(f"Fetching {TICKER} from Yahoo Finance...")
        prices, real_next = _fetch_yfinance(TICKER)
    elif MODE == "csv":
        prices, real_next = _fetch_csv(TICKER)
    elif MODE == "manual":
        assert len(MY_PRICES) == SEQ_LEN, f"MY_PRICES must have {SEQ_LEN} values"
        prices, real_next = MY_PRICES, None
    else:
        raise ValueError(f"Unknown MODE '{MODE}'")

    pred = _infer(model, prices)

    print(f"Ticker           : {TICKER if MODE != 'manual' else 'custom'}")
    print(f"Last known price : {prices[-1]:.2f}")
    print(f"Predicted next   : {pred:.2f}")
    if real_next is not None:
        print(f"Real next price  : {real_next:.2f}  (error: {abs(pred - real_next):.2f})")

    _plot(prices, pred, real_next)
    return pred


def _plot(prices: list, pred: float, real_next):
    import matplotlib.pyplot as plt

    n, x_pred = len(prices), len(prices) + 1

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(range(1, n + 1), prices, color="#4C9BE8", linewidth=2,
            marker="o", markersize=4, label="Input prices (last 30 days)")
    ax.plot([n, x_pred], [prices[-1], pred],
            color="#AAAAAA", linewidth=1.5, linestyle="--")
    ax.axvspan(n + 0.5, x_pred + 0.6, color="#FFEEEE", alpha=0.5)
    ax.axvline(x=n + 0.5, color="#DDDDDD", linewidth=1, linestyle=":")

    ax.scatter([x_pred], [pred], color="#E85C4C", s=160, zorder=6,
               label=f"Predicted: {pred:.2f}")
    ax.annotate(f"  {pred:.2f}", xy=(x_pred, pred), fontsize=11,
                color="#E85C4C", va="center", fontweight="bold")

    if real_next is not None:
        ax.scatter([x_pred], [real_next], color="#2ECC71", s=160,
                   zorder=6, marker="D", label=f"Real next: {real_next:.2f}")
        ax.annotate(f"  {real_next:.2f}", xy=(x_pred, real_next), fontsize=11,
                    color="#2ECC71", va="center", fontweight="bold")
        ax.annotate("", xy=(x_pred, real_next), xytext=(x_pred, pred),
                    arrowprops=dict(arrowstyle="<->", color="#999999", lw=1.5))
        ax.text(x_pred + 0.15, (pred + real_next) / 2,
                f"Δ {abs(pred - real_next):.2f}", fontsize=9, color="#888888", va="center")

    title = f"LSTM Stock Price Prediction — {TICKER if MODE != 'manual' else 'custom'}"
    if MODE == "yfinance":
        title += "  (live data)"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Day")
    ax.set_ylabel("Close Price")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("inference_plot.png", dpi=150)
    print("Plot saved → inference_plot.png")
    plt.show()


if __name__ == "__main__":
    run()
