import yfinance as yf
import pandas as pd


# fetch option chain data for a given stock symbol
def fetch_option_chain(symbol: str) -> pd.DataFrame:
    try:
        stock = yf.Ticker(symbol)
        options = stock.options
        if not options:
            return pd.DataFrame()  # No options available

        # Fetch option chain data for the nearest expiration date
        option_chain = stock.option_chain(options[0])
        calls = option_chain.calls
        puts = option_chain.puts
        

        # columns :['contractSymbol', 'lastTradeDate', 'strike', 'lastPrice', 'bid', 'ask', 'change', 'percentChange', 'volume', 'openInterest', 'impliedVolatility', 'inTheMoney', 'contractSize', 'currency', 'type']
        # Combine calls and puts into a single DataFrame
        calls['type'] = 'call'
        puts['type'] = 'put'
        option_data = pd.concat([calls, puts], ignore_index=True)

        return option_data
    except Exception as e:
        print(f"Error fetching option chain for {symbol}: {e}")
        return pd.DataFrame()
    
# lets check if the function is working or not
op = fetch_option_chain("AAPL")
print(op.columns.to_list())  # Example usage

