# Cryptocurrency Dashboard using Streamlit and CoinGecko API
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import sqlite3

conn = sqlite3.connect('crypto_daily_data.db')
cursor = conn.cursor()

def create_dynamic_table():
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS crypto_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin_id TEXT,
            date DATE,
            price REAL,
            market_cap REAL,
            volume REAL,
            UNIQUE(coin_id, date)
        )
    ''')
    conn.commit()

st.set_page_config(page_title="Cryptocurrency", page_icon="🪙")

st.title("🪙 Cryptocurrency Dashboard")

# Page configuration
st.set_page_config(
    page_title="Crypto Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .crypto-symbol {
        font-weight: bold;
        color: #ff7f0e;
    }
</style>
""", unsafe_allow_html=True)

# Cache functions for API calls
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_crypto_data(limit=10):
    """Fetch cryptocurrency data from CoinGecko API"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': limit,
            'page': 1,
            'sparkline': False,
            'price_change_percentage': '1h,24h,7d'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return []

@st.cache_data(ttl=300)
def fetch_historical_data(coin_id, days=30):
    """Fetch historical price data for a specific cryptocurrency"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily' if days > 30 else 'hourly'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Error fetching historical data: {str(e)}")
        return pd.DataFrame()

def format_number(num):
    """Format large numbers with appropriate suffixes"""
    if num >= 1e12:
        return f"${num/1e12:.2f}T"
    elif num >= 1e9:
        return f"${num/1e9:.2f}B"
    elif num >= 1e6:
        return f"${num/1e6:.2f}M"
    elif num >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"

def format_price(price):
    """Format price based on its value"""
    if price >= 1:
        return f"${price:,.2f}"
    else:
        return f"${price:.6f}"

def color_negative_red(val):
    """Color negative values red and positive values green"""
    if isinstance(val, str):
        return val
    color = 'red' if val < 0 else 'green'
    return f'color: {color}'

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Cryptocurrency Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        time.sleep(300)
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Number of cryptocurrencies to display
    num_coins = st.sidebar.slider("Number of coins to display", 10, 100, 20)
    
    # Fetch data
    with st.spinner("Fetching cryptocurrency data..."):
        crypto_data = fetch_crypto_data(num_coins)
    
    if not crypto_data:
        st.error("Failed to fetch cryptocurrency data. Please try again later.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(crypto_data)
    
    # Main metrics
    st.subheader("📊 Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_market_cap = df['market_cap'].sum()
        st.metric(
            label="Total Market Cap",
            value=format_number(total_market_cap)
        )
    
    with col2:
        total_volume = df['total_volume'].sum()
        st.metric(
            label="24h Trading Volume",
            value=format_number(total_volume)
        )
    
    with col3:
        avg_change_24h = df['price_change_percentage_24h'].mean()
        st.metric(
            label="Average 24h Change",
            value=f"{avg_change_24h:.2f}%",
            delta=f"{avg_change_24h:.2f}%"
        )
    
    with col4:
        btc_dominance = (df[df['symbol'] == 'btc']['market_cap'].iloc[0] / total_market_cap) * 100
        st.metric(
            label="BTC Dominance",
            value=f"{btc_dominance:.1f}%"
        )
    
    # Top cryptocurrencies table
    st.subheader("💰 Top Cryptocurrencies")
    
    # Prepare display DataFrame
    display_df = df[[
        'market_cap_rank', 'name', 'symbol', 'current_price', 
        'price_change_percentage_1h_in_currency',
        'price_change_percentage_24h', 'price_change_percentage_7d_in_currency',
        'market_cap', 'total_volume'
    ]].copy()
    
    display_df.columns = [
        'Rank', 'Name', 'Symbol', 'Price', 
        '1h %', '24h %', '7d %', 'Market Cap', '24h Volume'
    ]
    
    # Format the DataFrame
    display_df['Price'] = display_df['Price'].apply(format_price)
    display_df['Market Cap'] = display_df['Market Cap'].apply(format_number)
    display_df['24h Volume'] = display_df['24h Volume'].apply(format_number)
    display_df['Symbol'] = display_df['Symbol'].str.upper()
    
    # Style the DataFrame
    styled_df = display_df.style.applymap(
        color_negative_red, 
        subset=['1h %', '24h %', '7d %']
    ).format({
        '1h %': '{:.2f}%',
        '24h %': '{:.2f}%',
        '7d %': '{:.2f}%'
    })
    
    st.dataframe(styled_df, use_container_width=True, height=600)
    
    # Interactive charts
    st.subheader("📈 Market Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Market cap pie chart (top 10)
        top_10 = df.head(10)
        fig_pie = px.pie(
            values=top_10['market_cap'],
            names=top_10['name'],
            title="Market Cap Distribution (Top 10)",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # 24h change bar chart (top 20)
        top_20 = df.head(20)
        colors = ['red' if x < 0 else 'green' for x in top_20['price_change_percentage_24h']]
        
        fig_bar = go.Figure(data=[
            go.Bar(
                x=top_20['symbol'].str.upper(),
                y=top_20['price_change_percentage_24h'],
                marker_color=colors,
                text=top_20['price_change_percentage_24h'].round(2),
                textposition='outside'
            )
        ])
        fig_bar.update_layout(
            title="24h Price Change % (Top 20)",
            xaxis_title="Cryptocurrency",
            yaxis_title="Price Change %",
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Individual coin analysis
    st.subheader("🔍 Individual Coin Analysis")
    
    # Coin selector
    coin_names = {coin['name']: coin['id'] for coin in crypto_data}
    selected_coin_name = st.selectbox(
        "Select a cryptocurrency for detailed analysis:",
        options=list(coin_names.keys()),
        index=0
    )
    
    selected_coin_id = coin_names[selected_coin_name]
    selected_coin_data = next(coin for coin in crypto_data if coin['id'] == selected_coin_id)
    
    # Coin details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label=f"{selected_coin_data['name']} Price",
            value=format_price(selected_coin_data['current_price']),
            delta=f"{selected_coin_data['price_change_percentage_24h']:.2f}%"
        )
    
    with col2:
        st.metric(
            label="Market Cap Rank",
            value=f"#{selected_coin_data['market_cap_rank']}"
        )
    
    with col3:
        st.metric(
            label="Market Cap",
            value=format_number(selected_coin_data['market_cap'])
        )
    
    # Historical price chart
    time_periods = {
        "7 days": 7,
        "30 days": 30,
        "90 days": 90,
        "1 year": 365
    }
    
    selected_period = st.selectbox("Select time period:", list(time_periods.keys()))
    days = time_periods[selected_period]
    
    with st.spinner(f"Loading {selected_period} historical data..."):
        historical_df = fetch_historical_data(selected_coin_id, days)
    
    if not historical_df.empty:
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=historical_df['date'],
            y=historical_df['price'],
            mode='lines',
            name='Price',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig_line.update_layout(
            title=f"{selected_coin_data['name']} Price History ({selected_period})",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_line, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "📡 **Data provided by CoinGecko API** | "
        "⚡ **Auto-refresh available** | "
        f"🕒 **Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**"
    )

if __name__ == "__main__":
    main()