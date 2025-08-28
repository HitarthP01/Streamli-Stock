"""
this is version 3 of the sec_data app.
"""


import yfinance as yf
import pandas as pd
import datetime
import sqlite3
import streamlit as st
import ollama
from pandas_ollama import MyPandasAI
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def print_data(df: pd.DataFrame):

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    st.dataframe(df)
    st.plotly_chart(plot_price_volume(df), use_container_width=True)


def plot_price_volume(df: pd.DataFrame):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    # moving averages for context
    df['ma3'] = df['close'].rolling(3, min_periods=1).mean()
    df['ma5'] = df['close'].rolling(5, min_periods=1).mean()

    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        row_heights=[0.7, 0.3],
                        vertical_spacing=0.03)

    # If OHLC is available show candle, otherwise line
    if {'open', 'high', 'low'}.issubset(df.columns):
        fig.add_trace(go.Candlestick(x=df['date'],
                                     open=df['open'],
                                     high=df['high'],
                                     low=df['low'],
                                     close=df['close'],
                                     name='OHLC'), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=df['date'], y=df['close'],
                                 mode='lines+markers', name='Close'), row=1, col=1)

    # Add moving averages
    fig.add_trace(go.Scatter(x=df['date'], y=df['ma3'],
                             mode='lines', name='MA3', line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['ma5'],
                             mode='lines', name='MA5', line=dict(width=1)), row=1, col=1)

    # Volume as bars
    if 'volume' in df.columns:
        fig.add_trace(go.Bar(x=df['date'], y=df['volume'],
                             name='Volume', marker_color='lightgray'), row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False,
                      height=600,
                      margin=dict(l=20, r=20, t=30, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    return fig

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
    st.plotly_chart(plot_price_volume(data), use_container_width=True)

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
    st.plotly_chart(plot_price_volume(data), use_container_width=True)


# show data from SQLite using streamlit:
query = """SELECT DISTINCT symbol FROM prices WHERE symbol = ? ORDER BY symbol """
cursor.execute(query, (symbol,))
symbols = cursor.fetchall()


panoll = MyPandasAI(data, model="llama3:latest")

# get input from user in streamlit and ask it to the model
st.subheader("Ask a question about this data:")
question = st.text_input("Your question:")  
result = panoll.ask(question)
st.write(result.content)
