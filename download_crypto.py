
import yfinance as yf

# Download Bitcoin and Ethereum data
btc = yf.download("BTC-USD", start="2016-01-01")
eth = yf.download("ETH-USD", start="2016-01-01")

# Optional: Save to CSV files in the same folder
btc.to_csv("bitcoin_data.csv")
eth.to_csv("ethereum_data.csv")

print("✅ Data downloaded and saved!")

# this module is created to get ETH and BTC dataset from YAHOO Finance
