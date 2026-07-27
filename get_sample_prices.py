from data import load_data, SEQ_LEN, TRAIN_RATIO

CSV_PATH = "all_stocks_5yr.csv"
TICKER   = "AMT"
OFFSET   = 10  # shift window back from the end of test set

raw   = load_data(CSV_PATH, TICKER).flatten()
split = int(len(raw) * TRAIN_RATIO)
test  = raw[split:]
end   = len(test) - OFFSET
prices = test[end - SEQ_LEN : end].tolist()
real_next = float(test[end]) if end < len(test) else None

if real_next:
    print(f"Real next price: {real_next:.2f}\n")
print("MY_PRICES = [")
for row in [prices[i:i+10] for i in range(0, len(prices), 10)]:
    print("    " + ", ".join(f"{p:.2f}" for p in row) + ",")
print("]")
