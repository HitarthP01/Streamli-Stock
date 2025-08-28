import os
import yfinance as yf
import pandas as pd
import sqlite3
import streamlit as st
from pandas_ollama import MyPandasAI
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests

st.title("SEC Data App v3 - Top 20 Most Active Stocks")

# --------- DATABASE SETUP -----------
DB_FILENAME = "stocks_tp.db"
conn = sqlite3.connect(DB_FILENAME)
cursor = conn.cursor()

# Create tables if needed
# cursor.execute("""
# CREATE TABLE IF NOT EXISTS most_active_stocks (
#     Symbol TEXT,
#     Name TEXT,
#     Price_Intraday TEXT,
#     Change TEXT,
#     ChangePercent TEXT,
#     Volume TEXT,
#     Avg_Vol_3M TEXT,
#     Market_Cap TEXT,
#     PE_Ratio_TTM TEXT,
#     _52_Week_Range TEXT,
#     Region TEXT,
#     Scrape_Date TEXT)
# """)
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
cursor.execute("""
CREATE TABLE IF NOT EXISTS symbols (
    symbol TEXT,
    first_fetched TIMESTAMP,
    last_fetched TIMESTAMP
)
""")
conn.commit()


# --- SCRAPE + STORE AS DATAFRAME ---
url = 'https://finance.yahoo.com/research-hub/screener/most_actives/'
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

table = soup.find('table')
header_cells = table.find('thead').find_all('th')
columns = [h.text.replace(' ', '_').replace('%', 'Percent') for h in header_cells] + ['Scrape_Date']
rows = table.find('tbody').find_all('tr')

today = datetime.now().strftime('%Y-%m-%d')
data = []
for row in rows:
    cells = [td.text for td in row.find_all('td')]
    if cells:
        cells.append(today)
        data.append(cells)

# Turn into DataFrame
most_active_df = pd.DataFrame(data, columns=columns)
most_active_df['_Symbol__'] = most_active_df['_Symbol__'].str.split().str[-1]


print(most_active_df['_Symbol__'])

# Write to SQLite (replace to always get today’s latest)
most_active_df.to_sql("most_active_stocks", conn, if_exists="replace", index=False)

# --- READ BACK & DISPLAY TOP 20 ---
def parse_volume(volume_str):
    try:
        if volume_str[-1] in "MBK":
            mult = {'M': 1e6, 'B': 1e9, 'K': 1e3}[volume_str[-1]]
            return float(volume_str[:-1].replace(',', '')) * mult
        return float(volume_str.replace(',', ''))
    except:
        return 0

stocks_df = pd.read_sql_query(
    "SELECT * FROM most_active_stocks WHERE Scrape_Date = ?",
    conn,
    params=(today,)
)
stocks_df['VolumeAmount'] = stocks_df['_Volume__'].apply(parse_volume)
# stocks_df = stocks_df.sort_values('VolumeAmount', ascending=False).head(20)

st.subheader("Top 20 Most Active Stocks (by Volume)")
st.dataframe(stocks_df.drop('VolumeAmount', axis=1), hide_index=True)

# -------------- UI: Let user select ---------------------
choose_symbol = st.selectbox(
    "Select a stock symbol to view 7-day history:",
    options=stocks_df['_Symbol__'].tolist(),
    index=0
).upper()

# --------- Functions for plotting ---------
def plot_price_volume(df: pd.DataFrame):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df['ma3'] = df['close'].rolling(3, min_periods=1).mean()
    df['ma5'] = df['close'].rolling(5, min_periods=1).mean()
    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        row_heights=[0.7, 0.3],
                        vertical_spacing=0.03)
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
    fig.add_trace(go.Scatter(x=df['date'], y=df['ma3'],
                             mode='lines', name='MA3', line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['ma5'],
                             mode='lines', name='MA5', line=dict(width=1)), row=1, col=1)
    if 'volume' in df.columns:
        fig.add_trace(go.Bar(x=df['date'], y=df['volume'],
                             name='Volume', marker_color='lightgray'), row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False,
                      height=600,
                      margin=dict(l=20, r=20, t=30, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# --------- Show price/volume history ---------
# First, see if it's in db
query = "SELECT * FROM prices WHERE symbol = ? ORDER BY date DESC LIMIT 7"
data = pd.read_sql_query(query, conn, params=(choose_symbol.lower(),))

if not data.empty:
    st.subheader(f"Last 7 Trading Days for {choose_symbol}")
    st.dataframe(data)
    st.plotly_chart(plot_price_volume(data), use_container_width=True)
else:
    # Fetch from Yahoo Finance if not in db
    st.warning(f"{choose_symbol} not found in database. Fetching from Yahoo Finance...")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=14)
    data = yf.download(choose_symbol, start=start_date, end=end_date)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if col[0] else col[1] for col in data.columns]
    data = data.tail(7).reset_index()
    data.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume"
    }, inplace=True)
    data["symbol"] = choose_symbol.lower()
    expected_cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
    data = data[[c for c in expected_cols if c in data.columns]]
    # Save to db
    data.to_sql("prices", conn, if_exists="append", index=False)
    st.success(f"Data for {choose_symbol} fetched and saved to database.")
    st.subheader(f"Last 7 Trading Days for {choose_symbol}")
    st.dataframe(data)
    st.plotly_chart(plot_price_volume(data), use_container_width=True)

# --------- pandas-ai (Ollama) Chatbot -----------
st.subheader("Ask a question about this data:")
if not data.empty:
    panoll = MyPandasAI(data, model="llama3:latest")
    question = st.text_input("Your question:")
    if question:
        result = panoll.ask(question)
        st.write(result.content)

# ------- End -------

