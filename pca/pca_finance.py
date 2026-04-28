import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

from pca import PCA

df = pd.read_csv("state.x77.csv")
df = df.drop(columns=["rownames"])
states = df.columns.tolist()
st77 = df.to_numpy()
print(st77)

# Define the portfolio
#"MSFT", "AAPL", "GOOGL", "AMZN",
# "ASML.AS", "SAP.DE", "OR.PA","NESN.SW"
# "TSM", "005930.KS", "0700.HK", "TM"
symbols = ["^GSPC","MSFT", "AAPL", "GOOGL", "AMZN",
           "^STOXX50E","ASML.AS", "SAP.DE", "OR.PA","NESN.SW",
           "000001.SS","TSM", "005930.KS", "0700.HK", "TM"]

# 1. Fetch real data
end_date = "2022-01-01"
start_date = "2020-01-01"
data = yf.download(symbols, start=start_date, end=end_date)["Close"]

# 2. Compute daily returns
returns = data.pct_change().dropna()

returns_array = returns.to_numpy()

pca = PCA(returns_array)
pca.fit()
pca.summary()
projection = pca.transform(returns_array, 2)
print(projection)
plt.figure('Ticker evolution (2020-2021)')
plt.plot(projection,'.r')
plt.title('Ticker evolution (2020-2021)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
pca.biplot(returns_array,5,symbols)