import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("../data/AAPL.csv")


df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df.dropna(subset=["Date", "Close"], inplace=True)

df["MA_7"] = df["Close"].rolling(7).mean()
df["MA_30"] = df["Close"].rolling(30).mean()

plt.plot(df["Date"], df["Close"], label="Close")
plt.plot(df["Date"], df["MA_7"], label="MA_7")
plt.plot(df["Date"], df["MA_30"], label="MA_30")
plt.title("AAPL Close Price with Moving Averages")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

