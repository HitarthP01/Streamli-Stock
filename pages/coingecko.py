import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import time

# Page configuration
st.set_page_config(
    page_title="Crypto Historical Analysis",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Data fetching functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_crypto_data_coingecko(symbol, start_date, end_date):
    """Fetch cryptocurrency data from CoinGecko API"""
    
    # Map symbols to CoinGecko IDs
    coingecko_ids = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'ADA': 'cardano',
        'DOT': 'polkadot',
        'LINK': 'chainlink',
        'LTC': 'litecoin',
        'XRP': 'ripple',
        'BCH': 'bitcoin-cash',
        'BNB': 'binancecoin',
        'SOL': 'solana'
    }
    
    if symbol not in coingecko_ids:
        st.error(f"Symbol {symbol} not supported")
        return pd.DataFrame()
    
    coin_id = coingecko_ids[symbol]
    
    try:
        # Convert dates to timestamps
        start_timestamp = int(pd.Timestamp(start_date).timestamp())
        end_timestamp = int(pd.Timestamp(end_date).timestamp())
        
        # CoinGecko API endpoint for historical data
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
        params = {
            'vs_currency': 'usd',
            'from': start_timestamp,
            'to': end_timestamp
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if not data or 'prices' not in data:
            st.error(f"No data found for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        prices = data['prices']
        volumes = data['total_volumes']
        
        df = pd.DataFrame(prices, columns=['timestamp', 'Close'])
        df['Volume'] = [v[1] for v in volumes]
        
        # Convert timestamp to datetime
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('Date', inplace=True)
        df.drop('timestamp', axis=1, inplace=True)
        
        # Add OHLC approximations (CoinGecko free API only provides close prices)
        df['Open'] = df['Close'].shift(1).fillna(df['Close'])
        df['High'] = df['Close'] * 1.001  # Small approximation
        df['Low'] = df['Close'] * 0.999   # Small approximation
        df['Adj Close'] = df['Close']
        
        # Reorder columns to match Yahoo Finance format
        df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching data from CoinGecko: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache for 1 hour  
def fetch_crypto_data_yahoo(symbol, start_date, end_date):
    """Fetch cryptocurrency data from Yahoo Finance (fallback)"""
    
    try:
        # Map common crypto symbols to Yahoo Finance format
        yf_symbol = f"{symbol}-USD" if not symbol.endswith('-USD') else symbol
        
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            return pd.DataFrame()
        
        return df
        
    except Exception as e:
        return pd.DataFrame()

def fetch_crypto_data(symbol, start_date, end_date, data_source="auto"):
    """Fetch cryptocurrency data with multiple source support"""
    
    if data_source == "coingecko":
        return fetch_crypto_data_coingecko(symbol, start_date, end_date)
    elif data_source == "yahoo":
        return fetch_crypto_data_yahoo(symbol, start_date, end_date)
    else:  # auto
        # Try CoinGecko first (better historical coverage)
        df = fetch_crypto_data_coingecko(symbol, start_date, end_date)
        if not df.empty:
            st.info("✅ Data fetched from CoinGecko API (better historical coverage)")
            return df
        
        # Fallback to Yahoo Finance
        st.warning("🔄 CoinGecko failed, trying Yahoo Finance...")
        df = fetch_crypto_data_yahoo(symbol, start_date, end_date)
        if not df.empty:
            st.info("✅ Data fetched from Yahoo Finance")
            return df
        
        st.error("❌ Failed to fetch data from both sources")
        return pd.DataFrame()

def calculate_monthly_returns(df):
    """Calculate monthly percentage changes"""
    if df.empty:
        return pd.DataFrame()
    
    # Resample to monthly data (using last day of each month)
    monthly_data = df['Close'].resample('M').last()
    
    # Calculate percentage change
    monthly_returns = monthly_data.pct_change() * 100
    monthly_returns = monthly_returns.dropna()
    
    # Create a DataFrame with year and month
    monthly_df = pd.DataFrame({
        'Date': monthly_returns.index,
        'Return': monthly_returns.values
    })
    
    monthly_df['Year'] = monthly_df['Date'].dt.year
    monthly_df['Month'] = monthly_df['Date'].dt.month
    monthly_df['Month_Name'] = monthly_df['Date'].dt.strftime('%B')
    
    return monthly_df

def create_correlation_matrix(monthly_df):
    """Create correlation matrix for monthly returns across years"""
    if monthly_df.empty:
        return pd.DataFrame()
    
    # Pivot the data to have years as columns and months as rows
    pivot_df = monthly_df.pivot_table(
        values='Return', 
        index='Month', 
        columns='Year', 
        aggfunc='first'
    )
    
    return pivot_df

def create_year_correlation(monthly_df):
    """Create year-to-year correlation matrix"""
    pivot_df = create_correlation_matrix(monthly_df)
    if pivot_df.empty:
        return pd.DataFrame()
    
    # Calculate correlation between years
    year_corr = pivot_df.corr()
    
    return year_corr

# Visualization functions
def plot_monthly_returns_heatmap(monthly_df):
    """Create heatmap of monthly returns by year"""
    pivot_df = create_correlation_matrix(monthly_df)
    
    if pivot_df.empty:
        return None
    
    # Create month names for better display
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=[str(col) for col in pivot_df.columns],
        y=[month_names[idx-1] if idx <= 12 else f"Month {idx}" for idx in pivot_df.index],
        colorscale='RdYlGn',
        zmid=0,
        colorbar=dict(title="Monthly Return (%)", titleside="right"),
        hoverongaps=False,
        hovertemplate='Year: %{x}<br>Month: %{y}<br>Return: %{z:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Monthly Returns Heatmap by Year",
        xaxis_title="Year",
        yaxis_title="Month",
        height=500,
        font=dict(size=12)
    )
    
    return fig

def plot_year_correlation_heatmap(year_corr):
    """Create heatmap of year-to-year correlations"""
    if year_corr.empty:
        return None
    
    # Create annotation text for correlation values
    annotation_text = [[f"{val:.3f}" for val in row] for row in year_corr.values]
    
    fig = go.Figure(data=go.Heatmap(
        z=year_corr.values,
        x=[str(col) for col in year_corr.columns],
        y=[str(idx) for idx in year_corr.index],
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlation Coefficient", titleside="right"),
        text=annotation_text,
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate='Year 1: %{y}<br>Year 2: %{x}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Year-to-Year Correlation Matrix of Monthly Returns",
        xaxis_title="Year",
        yaxis_title="Year",
        height=600,
        font=dict(size=12)
    )
    
    return fig

def plot_monthly_returns_by_year(monthly_df):
    """Create line plot of monthly returns by year"""
    if monthly_df.empty:
        return None
    
    fig = px.line(
        monthly_df, 
        x='Month', 
        y='Return', 
        color='Year',
        title="Monthly Returns by Year",
        labels={'Return': 'Monthly Return (%)', 'Month': 'Month'},
        hover_data=['Month_Name']
    )
    
    fig.update_layout(
        height=500,
        hovermode='x unified'
    )
    fig.update_xaxis(
        tickmode='array',
        tickvals=list(range(1, 13)),
        ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    )
    
    return fig

def plot_cumulative_returns(df):
    """Plot cumulative returns over time"""
    if df.empty:
        return None
    
    # Calculate cumulative returns
    daily_returns = df['Close'].pct_change()
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=cumulative_returns * 100,
        mode='lines',
        name='Cumulative Returns',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='Date: %{x}<br>Cumulative Return: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Cumulative Returns Over Time",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        height=400,
        showlegend=False
    )
    
    return fig

def plot_monthly_distribution(monthly_df):
    """Plot distribution of monthly returns"""
    if monthly_df.empty:
        return None
    
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=monthly_df['Return'],
        nbinsx=30,
        name='Monthly Returns',
        opacity=0.7,
        marker_color='lightblue'
    ))
    
    # Add vertical line for mean
    mean_return = monthly_df['Return'].mean()
    fig.add_vline(
        x=mean_return,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_return:.2f}%"
    )
    
    fig.update_layout(
        title="Distribution of Monthly Returns",
        xaxis_title="Monthly Return (%)",
        yaxis_title="Frequency",
        height=400
    )
    
    return fig

def calculate_risk_metrics(monthly_df):
    """Calculate risk and performance metrics"""
    if monthly_df.empty:
        return {}
    
    returns = monthly_df['Return'].values
    
    metrics = {
        'Mean Monthly Return': f"{np.mean(returns):.2f}%",
        'Monthly Volatility': f"{np.std(returns):.2f}%",
        'Annualized Return': f"{np.mean(returns) * 12:.2f}%",
        'Annualized Volatility': f"{np.std(returns) * np.sqrt(12):.2f}%",
        'Sharpe Ratio (0% risk-free)': f"{(np.mean(returns) * 12) / (np.std(returns) * np.sqrt(12)):.3f}",
        'Best Month': f"{np.max(returns):.2f}%",
        'Worst Month': f"{np.min(returns):.2f}%",
        'Positive Months': f"{len(returns[returns > 0])}/{len(returns)} ({len(returns[returns > 0])/len(returns)*100:.1f}%)"
    }
    
    return metrics

# Main application
def main():
    st.title("₿ Cryptocurrency Historical Analysis")
    st.markdown("Analyze historical cryptocurrency performance with monthly correlation studies")
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    # Cryptocurrency selection
    crypto_options = {
        'Bitcoin (BTC)': 'BTC',
        'Ethereum (ETH)': 'ETH',
        'Cardano (ADA)': 'ADA',
        'Polkadot (DOT)': 'DOT',
        'Chainlink (LINK)': 'LINK',
        'Litecoin (LTC)': 'LTC',
        'XRP (XRP)': 'XRP',
        'Bitcoin Cash (BCH)': 'BCH',
        'Binance Coin (BNB)': 'BNB',
        'Solana (SOL)': 'SOL'
    }
    
    selected_display = st.sidebar.selectbox("Select Cryptocurrency", list(crypto_options.keys()), index=0)
    selected_crypto = crypto_options[selected_display]
    
    # Date range selection
    st.sidebar.subheader("Date Range")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*12)  # 12 years ago
    
    start_date = st.sidebar.date_input("Start Date", start_date)
    end_date = st.sidebar.date_input("End Date", end_date)
    
    # Data source selection
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["auto", "coingecko", "yahoo"],
        index=0,
        help="Auto: Tries CoinGecko first, then Yahoo Finance\nCoinGecko: Better historical data\nYahoo: Recent data only"
    )
    # Analysis options
    st.sidebar.subheader("Analysis Options")
    show_distribution = st.sidebar.checkbox("Show Return Distribution", True)
    show_risk_metrics = st.sidebar.checkbox("Show Risk Metrics", True)
    
    # Control buttons
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()
    
    if start_date >= end_date:
        st.error("Start date must be before end date!")
        return
    
    # Fetch and process data
    with st.spinner(f"Fetching {selected_display} data..."):
        df = fetch_crypto_data(selected_crypto, start_date, end_date, data_source)
    
    if df.empty:
        st.error("No data available for the selected cryptocurrency and date range.")
        return
    
    # Calculate monthly returns
    monthly_df = calculate_monthly_returns(df)
    
    if monthly_df.empty:
        st.error("Insufficient data to calculate monthly returns.")
        return
    
    # Display basic statistics
    st.header("📊 Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        years_span = monthly_df['Year'].max() - monthly_df['Year'].min() + 1
        st.metric("Years of Data", f"{years_span}")
    with col3:
        avg_return = monthly_df['Return'].mean()
        st.metric("Avg Monthly Return", f"{avg_return:.2f}%")
    with col4:
        volatility = monthly_df['Return'].std()
        st.metric("Monthly Volatility", f"{volatility:.2f}%")
    
    # Price chart
    st.header("📈 Price History")
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name=f'{selected_crypto} Price',
        line=dict(color='orange', width=2),
        hovertemplate='Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
    ))
    price_fig.update_layout(
        title=f"{selected_display} Price History",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=400,
        showlegend=False
    )
    st.plotly_chart(price_fig, use_container_width=True)
    
    # Cumulative returns
    st.header("📊 Cumulative Returns")
    cum_returns_fig = plot_cumulative_returns(df)
    if cum_returns_fig:
        st.plotly_chart(cum_returns_fig, use_container_width=True)
    
    # Monthly returns analysis
    st.header("📅 Monthly Returns Analysis")
    
    # Monthly returns heatmap
    st.subheader("Monthly Returns by Year")
    monthly_heatmap = plot_monthly_returns_heatmap(monthly_df)
    if monthly_heatmap:
        st.plotly_chart(monthly_heatmap, use_container_width=True)
        st.markdown("*Green indicates positive returns, red indicates negative returns*")
    
    # Monthly returns by year line chart
    st.subheader("Monthly Returns Trends")
    monthly_line_fig = plot_monthly_returns_by_year(monthly_df)
    if monthly_line_fig:
        st.plotly_chart(monthly_line_fig, use_container_width=True)
    
    # Distribution plot
    if show_distribution:
        st.subheader("Monthly Returns Distribution")
        dist_fig = plot_monthly_distribution(monthly_df)
        if dist_fig:
            st.plotly_chart(dist_fig, use_container_width=True)
    
    # Correlation analysis
    st.header("🔗 Correlation Analysis")
    
    year_corr = create_year_correlation(monthly_df)
    if not year_corr.empty:
        # Year correlation heatmap
        st.subheader("Year-to-Year Correlation Matrix")
        st.markdown("*This shows how monthly return patterns correlate between different years*")
        corr_heatmap = plot_year_correlation_heatmap(year_corr)
        if corr_heatmap:
            st.plotly_chart(corr_heatmap, use_container_width=True)
        
        # Correlation statistics
        st.subheader("Correlation Insights")
        col1, col2, col3 = st.columns(3)
        
        # Calculate correlation statistics
        corr_values = year_corr.values
        mask = np.triu(np.ones_like(corr_values, dtype=bool), k=1)
        upper_triangle = corr_values[mask]
        
        with col1:
            avg_corr = upper_triangle.mean()
            st.metric("Average Correlation", f"{avg_corr:.3f}")
        
        with col2:
            max_corr = upper_triangle.max()
            st.metric("Highest Correlation", f"{max_corr:.3f}")
        
        with col3:
            min_corr = upper_triangle.min()
            st.metric("Lowest Correlation", f"{min_corr:.3f}")
    
    # Risk metrics
    if show_risk_metrics:
        st.header("📊 Risk & Performance Metrics")
        metrics = calculate_risk_metrics(monthly_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Return Metrics")
            st.metric("Mean Monthly Return", metrics['Mean Monthly Return'])
            st.metric("Annualized Return", metrics['Annualized Return'])
            st.metric("Best Month", metrics['Best Month'])
            st.metric("Worst Month", metrics['Worst Month'])
        
        with col2:
            st.subheader("Risk Metrics")
            st.metric("Monthly Volatility", metrics['Monthly Volatility'])
            st.metric("Annualized Volatility", metrics['Annualized Volatility'])
            st.metric("Sharpe Ratio", metrics['Sharpe Ratio (0% risk-free)'])
            st.metric("Positive Months", metrics['Positive Months'])
    
    # Monthly statistics table
    st.header("📋 Monthly Performance Summary")
    
    # Calculate monthly statistics
    monthly_stats = monthly_df.groupby('Month_Name')['Return'].agg([
        ('Mean', 'mean'),
        ('Std Dev', 'std'),
        ('Min', 'min'),
        ('Max', 'max'),
        ('Count', 'count')
    ]).round(2)
    
    # Reorder by month
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_stats = monthly_stats.reindex([m for m in month_order if m in monthly_stats.index])
    
    # Style the dataframe
    styled_df = monthly_stats.style.format("{:.2f}").background_gradient(
        subset=['Mean'], cmap='RdYlGn', vmin=-10, vmax=10
    )
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Export data section
    st.header("💾 Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Monthly Returns"):
            csv_data = monthly_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv_data,
                f"{selected_crypto}_monthly_returns_{start_date}_{end_date}.csv",
                "text/csv",
                key="monthly_returns"
            )
    
    with col2:
        if not year_corr.empty:
            if st.button("📥 Download Correlation Matrix"):
                corr_csv = year_corr.to_csv()
                st.download_button(
                    "Download CSV",
                    corr_csv,
                    f"{selected_crypto}_correlation_matrix_{start_date}_{end_date}.csv",
                    "text/csv",
                    key="correlation_matrix"
                )
    
    # Data info
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Info")
    
    if not df.empty:
        data_info = f"""
        **Data Source:** {data_source.title() if data_source != 'auto' else 'Auto (CoinGecko → Yahoo)'}
        **Symbol:** {selected_crypto}
        **Records:** {len(df):,} daily prices
        **Monthly Records:** {len(monthly_df) if not monthly_df.empty else 0}
        **Date Range:** {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}
        """
    else:
        data_info = "No data loaded"
    
    st.sidebar.info(data_info)
    
    # API Information
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Sources Info")
    st.sidebar.markdown("""
    **CoinGecko API:**
    - ✅ Data from 2013+ for Bitcoin
    - ✅ Free tier: 10-50 calls/min
    - ✅ Reliable historical data
    
    **Yahoo Finance:**
    - ⚠️ Limited crypto history
    - ✅ Good for recent data (2017+)
    - ✅ Full OHLCV data
    """)

if __name__ == "__main__":
    main()