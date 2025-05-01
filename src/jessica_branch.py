import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/jessica/Downloads/TSLA.csv", skiprows=2)

df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

df["Date"] = pd.to_datetime(df["Date"])
df["Close"] = pd.to_numeric(df["Close"], errors='coerce')

plt.figure(figsize=(14, 6))
plt.bar(df["Date"], df["Close"], width=1.5)
plt.xlabel("Date")
plt.ylabel("Close Price (USD)")
plt.title("TSLA Close Price Over Time")
plt.xticks(rotation=45)