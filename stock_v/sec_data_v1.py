"""Version 1.0.0
Fetches last 7 days of stock data from Yahoo Finance"""

import streamlit as st
import yfinance as yf
import pandas as pd
import datetime

# Title
st.title("7-Day Asset Data & Chatbot")

# 1. Asset Selection, in future this could be a dropdown or search
symbol = st.text_input("Enter a stock symbol (e.g., AAPL, TSLA):", "AAPL")

if symbol:
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=14)  # fetch more to account for weekends
    data = yf.download(symbol, start=start_date, end=end_date)
    data = data.tail(7)  # Get the last 7 rows
    
    if not data.empty:
        # 2. Show Data
        st.subheader(f"Last 7 Trading Days for {symbol.upper()}")
        st.dataframe(data)
        st.line_chart(data['Close'])

        # 3. User Q&A
        st.subheader("Ask a question about this data:")
        question = st.text_input("Your question:")
        
        if question:
            answer = ""
            # Basic example logic -- expand as needed
            if "highest" in question.lower() and "close" in question.lower():
                highest = data['Close'].max()
                answer = f"Highest closing price: {highest:.2f}"
            elif "lowest" in question.lower() and "close" in question.lower():
                lowest = data['Close'].min()
                answer = f"Lowest closing price: {lowest:.2f}"
            elif "average" in question.lower():
                avg = data['Close'].mean()
                answer = f"Average closing price: {avg:.2f}"
            else:
                answer = "Sorry, I didn't understand the question. Try asking for highest/lowest/average closing price."
            
            st.write(answer)
    else:
        st.error("No data found. Check the symbol and try again.")
