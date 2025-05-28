import pandas as pd
<<<<<<< HEAD
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
=======
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

np.random.seed(1)
n = 100
ma7 = np.random.normal(loc=100, scale=10, size=n)
ma30 = ma7 + np.random.normal(loc=0, scale=5, size=n)
noise = np.random.normal(loc=0, scale=2, size=n)
closing_price = 0.6 * ma7 + 0.3 * ma30 + noise

df = pd.DataFrame({
    'ma7': ma7,
    'ma30': ma30,
    'closing_price': closing_price
})

X_simple = df[['ma7']]
y = df['closing_price']

model_simple = LinearRegression()
model_simple.fit(X_simple, y)

X_multiple = df[['ma7', 'ma30']]
model_multiple = LinearRegression()
model_multiple.fit(X_multiple, y)

print("Simple Linear Regression:")
print("  R^2 Score:", model_simple.score(X_simple, y))
print("  Coefficient(s):", model_simple.coef_)
print("  Intercept:", model_simple.intercept_)
print()

print("Multiple Linear Regression:")
print("  R^2 Score:", model_multiple.score(X_multiple, y))
print("  Coefficient(s):", model_multiple.coef_)
print("  Intercept:", model_multiple.intercept_)
>>>>>>> ec12631 (Add Streamlit forecast summary metrics)

