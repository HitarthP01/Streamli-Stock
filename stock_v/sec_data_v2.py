"""
In this version, we fetch and display the last 7 days data using db and not from the API directly.
"""
import yfinance as yf
import pandas as pd
import datetime
import sqlite3
import streamlit as st


# Connect to SQLite (creates file if it doesn't exist)
conn = sqlite3.connect("stocks.db")
cursor = conn.cursor()

# Ensure the database and tables are created
cursor.execute("""
CREATE TABLE IF NOT EXISTS prices ( 
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    open REAL, 
    high REAL, 
    low REAL, 
    close REAL,  
    volume INTEGER,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
""")
    # PRIMARY KEY (symbol, date)

cursor.execute("""
CREATE TABLE IF NOT EXISTS symbols (    
    symbol TEXT,
    first_fetched TIMESTAMP,
    last_fetched TIMESTAMP
)
""")

conn.commit()



# Title
st.title("7-Day Asset Data & Chatbot")

# 1. Asset Selection, in future this could be a dropdown or search
symbol = st.text_input("Enter a stock symbol (e.g., AAPL, TSLA):", "aapl")
# make sure symbol is uppercase
symbol = symbol.lower()
print(symbol)



# 2. Check if symbol exists in the database and fetch data
cursor.execute("SELECT 1 FROM symbols WHERE symbol = ?", (symbol,))
exists = cursor.fetchone()



# if symbol exists in the database, fetch from there
if exists:
    # show data from SQLite using streamlit:
    query = "SELECT * FROM prices WHERE symbol = ? ORDER BY date DESC LIMIT 7"
    data = pd.read_sql_query(query, conn, params=(symbol,))
    st.subheader(f"Last 7 Trading Days for {symbol}")
    st.dataframe(data)
    st.line_chart(data['close'])
else:
    # If symbol does not exist, fetch from Yahoo Finance
    st.warning(f"Symbol {symbol} not found in database. Fetching from Yahoo Finance...")
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=14)  # fetch more to account for weekends
    # Fetch data
    data = yf.download(symbol, start=start_date, end=end_date)

    # Flatten MultiIndex columns if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if col[0] else col[1] for col in data.columns]

    # Reset index so Date becomes a column
    data = data.tail(7).reset_index()

    # Standardize column names
    data.rename(columns={
        "Date": "date", 
        "Open": "open", 
        "High": "high",
        "Low": "low", 
        "Close": "close", 
        "Adj Close": "adj_close",
        "Volume": "volume"
    }, inplace=True)

    # Add symbol column
    data["symbol"] = symbol

    # Select only the columns you need (and avoid KeyError if adj_close not present)
    expected_cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
    data = data[[c for c in expected_cols if c in data.columns]]
    # # Add a symbol column
    # print(data)
    # print(data.columns.tolist())

    # Write to SQLite
    data.to_sql("prices", conn, if_exists="append", index=False)

    # write symbol to symbols table
    # symbol_data = {"symbol": symbol.upper(), 
    #                "first_fetched": datetime.datetime.utcnow().isoformat(), 
    #                "last_fetched": datetime.datetime.utcnow().isoformat()}
    # symbol = pd.DataFrame([symbol_data])
    # # Write symbol to symbols table 
    # symbol.to_sql("symbols", conn, if_exists="replace", index=False)

    # insert or update symbol in symbols table
    now = datetime.datetime.utcnow().isoformat()
    cursor.execute("""
    INSERT INTO symbols(symbol, first_fetched, last_fetched)
    VALUES (?, ?, ?)
    """, (symbol, now, now))
    conn.commit()
    # conn.close()
    # Show success message

    st.success(f"Data for {symbol} fetched and saved to database.")
    st.subheader(f"Last 7 Trading Days for {symbol.upper()}")
    st.dataframe(data)
    st.line_chart(data['close'])

# show data from SQLite using streamlit:
query = """SELECT DISTINCT symbol FROM prices WHERE symbol = ? ORDER BY symbol """
cursor.execute(query, (symbol,))
symbols = cursor.fetchall()





# st.subheader("Available Symbols")
# if symbols:
#     st.write([s[0] for s in symbols])








"""

"""