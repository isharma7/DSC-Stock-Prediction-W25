import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/AAPL.csv')

print(df.head())

df['Date'] = pd.to_datetime(df['Date'])

plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Close'], label='AAPL Closing Price', color='blue')
plt.title('AAPL Closing Prices As Time Goes')
plt.xlabel('Date')
plt.ylabel('Prices ($)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Creates a small line graph that shows the closing prices