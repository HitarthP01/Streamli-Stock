"""Version 4.0.0--------------------------------"""

import os
import yfinance as yf
import pandas as pd
import sqlite3
import streamlit as st
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import time
import logging
from typing import Optional, List, Dict
import asyncio
import aiohttp


st.set_page_config(page_title="Active Stocks", page_icon="📈")

st.title("📈 Most Active Stocks")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Advanced Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stock-positive {
        color: #00ff00;
        font-weight: bold;
    }
    .stock-negative {
        color: #ff4444;
        font-weight: bold;
    }
    .data-table {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

class DatabaseManager:
    def __init__(self, db_name: str = "enhanced_stocks_4.db"):
        self.db_name = db_name
        self.init_database()
    
    def get_connection(self):
        return sqlite3.connect(self.db_name, check_same_thread=False)
    
    def init_database(self):
        """Initialize database with proper schema and indexes"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Most active stocks table with better schema
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS most_active_stocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                name TEXT,
                price_intraday REAL,
                change_amount REAL,
                change_percent REAL,
                volume INTEGER,
                avg_vol_3m INTEGER,
                market_cap TEXT,
                pe_ratio REAL,
                week_52_range TEXT,
                scrape_date DATE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, scrape_date)
            )
            """)
            
            # Enhanced prices table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume INTEGER,
                sma_20 REAL,
                sma_50 REAL,
                rsi REAL,
                UNIQUE(symbol, date)
            )
            """)
            
            # Symbols tracking table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS symbols_tracking (
                symbol TEXT PRIMARY KEY,
                first_fetched TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_fetched TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                fetch_count INTEGER DEFAULT 1,
                is_active BOOLEAN DEFAULT TRUE
            )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_stock_symbol_date ON stock_prices(symbol, date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_active_stocks_date ON most_active_stocks(scrape_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_active_stocks_symbol ON most_active_stocks(symbol)")
            
            conn.commit()

class StockDataFetcher:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    
           
# ...existing code...
    def fetch_most_active_stocks(_self) -> Optional[pd.DataFrame]:
        """Fetch most active stocks with better error handling"""
        try:
            url = 'https://finance.yahoo.com/research-hub/screener/most_actives/'
            
            with st.spinner("Fetching most active stocks..."):
                response = requests.get(url, headers=_self.headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                table = soup.find('table')
                
                if not table:
                    st.error("Could not find stock data table on the webpage")
                    return None
                
                # Extract headers
                header_cells = table.find('thead').find_all('th')
                columns = [h.get_text(strip=True) for h in header_cells] 
                # columns = [h.text.replace(' ', '_').replace('%', 'Percent') for h in header_cells] + ['Scrape_Date']
                # Extract data rows
                rows = table.find('tbody').find_all('tr')
                today = datetime.now().strftime('%Y-%m-%d')
                
                # data = []
                # for row in rows:
                #     cells = [td.text for td in row.find_all('td')]
                #     if cells:
                #         cells.append(today)
                #         data.append(cells)
                                
                data = []
                for row in rows:
                    # cells = [td.get_text(strip=True) for td in row.find_all('td')]
                    cells = [td.text for td in row.find_all('td')]
                    if len(cells) == len(columns):
                        data.append(cells)
                
                if not data:
                    st.warning("No stock data found")
                    return None
                
                df = pd.DataFrame(data, columns=columns)

                # # --- Save raw extracted table to CSV BEFORE any cleaning/processing ---
                # try:
                #     raw_dir = os.path.join(os.getcwd(), "raw_data")
                #     os.makedirs(raw_dir, exist_ok=True)
                #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                #     raw_path = os.path.join(raw_dir, f"most_active_raw_{timestamp}.csv")
                #     df.to_csv(raw_path, index=False)
                #     logger.info(f"Saved raw scraped CSV to {raw_path}")
                # except Exception as e:
                #     logger.error(f"Failed to save raw CSV: {e}")
                # --------------------------------------------------------------------

                df['scrape_date'] = datetime.now().strftime('%Y-%m-%d')
                
                
                # Clean and process data
                df = _self._clean_stock_data(df)
                # print(df.head())
                print(df.columns.tolist())
                
                # Save to database
                _self._save_active_stocks_to_db(df)
                
                return df
                
        except requests.RequestException as e:
            st.error(f"Network error while fetching data: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Error processing stock data: {str(e)}")
            return None
    
    def _clean_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize stock data"""
        # Standardize column names
        column_mapping = {
            'Symbol': 'symbol',
            'Name': 'name',
            'Price (Intraday)': 'price_intraday',
            'Change': 'change_amount',
            '% Change': 'change_percent',
            'Volume': 'volume',
            'Avg Vol (3 month)': 'avg_vol_3m',
            'Market Cap': 'market_cap',
            'PE Ratio (TTM)': 'pe_ratio',
            '52 Week Range': 'week_52_range'
        }
        
        # Rename columns that exist
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Clean symbol column (remove extra text)
        if 'symbol' in df.columns:
            df['symbol'] = df['symbol'].str.split().str[-1].str.upper()
        
        # Parse numeric columns
        numeric_columns = ['price_intraday', 'change_amount', 'change_percent', 'pe_ratio']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
        
        # Parse volume columns
        volume_columns = ['volume', 'avg_vol_3m']
        for col in volume_columns:
            if col in df.columns:
                df[col] = df[col].apply(self._parse_volume)
        
        return df
    
    def _parse_volume(self, volume_str: str) -> int:
        """Parse volume strings like '1.2M', '500K', etc."""
        try:
            if pd.isna(volume_str) or volume_str == '':
                return 0
            
            volume_str = str(volume_str).replace(',', '').upper()
            
            if volume_str.endswith('M'):
                return int(float(volume_str[:-1]) * 1_000_000)
            elif volume_str.endswith('B'):
                return int(float(volume_str[:-1]) * 1_000_000_000)
            elif volume_str.endswith('K'):
                return int(float(volume_str[:-1]) * 1_000)
            else:
                return int(float(volume_str))
        except:
            return 0
    
    # def _save_active_stocks_to_db(self, df: pd.DataFrame):
    #     """Save active stocks data to database"""
    #     try:
    #         with self.db_manager.get_connection() as conn:
    #             df.to_sql("most_active_stocks", conn, if_exists="replace", index=False)
    #     except Exception as e:
    #         logger.error(f"Error saving active stocks to database: {e}")
    

# ...existing code...
    def _sanitize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize and clean column names; drop empty columns."""
        df = df.copy()
        # Normalize header strings
        clean_cols = []
        for c in df.columns:
            if c is None:
                clean_cols.append("") 
                continue
            name = str(c).strip()
            # common replacements
            name = name.replace('%', 'percent')
            name = name.replace('/', '_')
            name = name.replace('(', '').replace(')', '')
            name = name.replace('-', '_')
            name = name.replace('.', '')
            # collapse spaces and make lowercase
            name = "_".join(name.split()).lower()
            # collapse multiple underscores
            while "__" in name:
                name = name.replace("__", "_")
            clean_cols.append(name)
        df.columns = clean_cols

        # drop truly empty column names
        df = df.loc[:, [c for c in df.columns if c != ""]]

        # map a few known variants to canonical names
        mapping = {
            '1d_chart': 'chart_1d',
            'price_intraday': 'price_intraday',
            'change_%': 'change_percent',
            'change_percent': 'change_percent',
            'change': 'change_amount',
            'avg_vol_3m': 'avg_vol_3m',
            'avg_vol_(3m)': 'avg_vol_3m',
            'p_e_ratio_ttm': 'pe_ratio',
            'p_e_ratio_ttm': 'pe_ratio',
            '52_week_range': 'week_52_range',
            'market_cap': 'market_cap'
        }
        df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

        # remove duplicate columns (keep first)
        df = df.loc[:, ~df.columns.duplicated()]

        return df

    def _save_active_stocks_to_db(self, df: pd.DataFrame):
        """Save active stocks to DB using INSERT OR REPLACE after sanitizing columns."""
        try:
            df = self._sanitize_columns(df)
            # ensure scrape_date column exists
            if 'scrape_date' not in df.columns:
                df['scrape_date'] = datetime.now().strftime('%Y-%m-%d')

            # desired DB columns (match most_active_stocks schema)
            cols = [
                'symbol', 'name', 'price_intraday', 'change_amount', 'change_percent',
                'volume', 'avg_vol_3m', 'market_cap', 'pe_ratio', 'week_52_range', 'scrape_date'
            ]
            # keep only available columns and preserve order, fill missing with None
            insert_cols = [c for c in cols]
            rows = []
            for _, row in df.iterrows():
                vals = []
                for c in insert_cols:
                    if c in df.columns:
                        v = row[c]
                        vals.append(None if (pd.isna(v) or v == '') else v)
                    else:
                        vals.append(None)
                rows.append(tuple(vals))

            if not rows:
                logger.error("No rows to insert for active stocks")
                return

            placeholders = ",".join(["?"] * len(insert_cols))
            col_list = ",".join(insert_cols)
            sql = f"INSERT OR REPLACE INTO most_active_stocks ({col_list}) VALUES ({placeholders})"

            with self.db_manager.get_connection() as conn:
                cur = conn.cursor()
                cur.executemany(sql, rows)
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving active stocks to database: {e}")

    def _save_price_data_to_db(self, df: pd.DataFrame):
        """Save price data to DB using INSERT OR IGNORE (avoid pandas 'method' param)."""
        try:
            # sanitize column names
            df = df.copy()
            df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]

            # required/target columns in stock_prices table
            required = ['symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'sma_20', 'sma_50', 'rsi']
            cols = [c for c in required if c in df.columns] + [c for c in df.columns if c not in required]

            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

            # Prepare rows
            data_tuples = [tuple(row) for row in df[cols].fillna(None).itertuples(index=False, name=None)]
            if not data_tuples:
                logger.info("No price rows to save")
                return

            placeholders = ",".join(["?"] * len(cols))
            col_list = ",".join(cols)
            sql = f"INSERT OR IGNORE INTO stock_prices ({col_list}) VALUES ({placeholders})"

            with self.db_manager.get_connection() as conn:
                cur = conn.cursor()
                cur.executemany(sql, data_tuples)
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving price data to database: {e}")
# ...existing code...

    def fetch_stock_history(self, symbol: str, period: str = "1mo") -> Optional[pd.DataFrame]:
        """Fetch historical stock data with technical indicators"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            if df.empty:
                return None
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Standardize column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            df['symbol'] = symbol.lower()
            
            # Calculate technical indicators
            df = self._calculate_technical_indicators(df)
            
            # Save to database
            self._save_price_data_to_db(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching stock history for {symbol}: {e}")
            return None
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _save_price_data_to_db(self, df: pd.DataFrame):
        """Save price data to database"""
        try:
            with self.db_manager.get_connection() as conn:
                df.to_sql("stock_prices", conn, if_exists="append", index=False, method='ignore')
        except Exception as e:
            logger.error(f"Error saving price data to database: {e}")

class StockVisualizer:
    @staticmethod
    def create_advanced_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
        """Create advanced candlestick chart with technical indicators"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.6, 0.2, 0.2],
            vertical_spacing=0.05,
            subplot_titles=('Price & Technical Indicators', 'Volume', 'RSI')
        )
        
        # Candlestick chart
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            fig.add_trace(
                go.Candlestick(
                    x=df['date'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='OHLC',
                    increasing_line_color='#00ff00',
                    decreasing_line_color='#ff4444'
                ),
                row=1, col=1
            )
        
        # Moving averages
        if 'sma_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'], y=df['sma_20'],
                    mode='lines', name='SMA 20',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        if 'sma_50' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'], y=df['sma_50'],
                    mode='lines', name='SMA 50',
                    line=dict(color='purple', width=1)
                ),
                row=1, col=1
            )
        
        # Volume
        if 'volume' in df.columns:
            colors = ['green' if close >= open else 'red' 
                     for close, open in zip(df['close'], df['open'])]
            fig.add_trace(
                go.Bar(
                    x=df['date'], y=df['volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # RSI
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'], y=df['rsi'],
                    mode='lines', name='RSI',
                    line=dict(color='blue', width=2)
                ),
                row=3, col=1
            )
            
            # RSI overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"{symbol.upper()} - Advanced Stock Chart",
            xaxis_rangeslider_visible=False,
            height=800,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
        
        return fig

def main():
    # Initialize components
    db_manager = DatabaseManager()
    data_fetcher = StockDataFetcher(db_manager)
    visualizer = StockVisualizer()
    
    # Header
    st.markdown('<h1 class="main-header">📈 Advanced Stock Market Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Dashboard Settings")
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (5 min)", value=False)
    if auto_refresh:
        time.sleep(300)  # 5 minutes
        st.rerun()
    
    # Manual refresh
    if st.sidebar.button("🔄 Refresh Data Now"):
        st.cache_data.clear()
        st.rerun()
    
    # Data source selection
    data_source = st.sidebar.radio(
        "📊 Data Source:",
        ["Live Data (Yahoo)", "Database Cache"]
    )
    
    # Fetch most active stocks
    try:
        if data_source == "Live Data (Yahoo)":
            stocks_df = data_fetcher.fetch_most_active_stocks()
        else:
            with db_manager.get_connection() as conn:
                today = datetime.now().strftime('%Y-%m-%d')
                stocks_df = pd.read_sql_query(
                    "SELECT * FROM most_active_stocks WHERE scrape_date = ? ORDER BY volume DESC",
                    conn,
                    params=(today,)
                )
        
        if stocks_df is None or stocks_df.empty:
            st.error("No stock data available. Please try refreshing or check your connection.")
            return
        
    except Exception as e:
        st.error(f"Error loading stock data: {str(e)}")
        return
    
    # Market Overview
    st.subheader("📊 Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_volume = stocks_df['volume'].sum() if 'volume' in stocks_df.columns else 0
        st.metric(
            label="Total Volume",
            value=f"{total_volume:,.0f}" if total_volume > 0 else "N/A"
        )
    
    with col2:
        avg_change = stocks_df['change_percent'].mean() if 'change_percent' in stocks_df.columns else 0
        st.metric(
            label="Average Change %",
            value=f"{avg_change:.2f}%" if not pd.isna(avg_change) else "N/A",
            delta=f"{avg_change:.2f}%" if not pd.isna(avg_change) else None
        )
    
    with col3:
        gainers = len(stocks_df[stocks_df['change_percent'] > 0]) if 'change_percent' in stocks_df.columns else 0
        st.metric(
            label="Gainers",
            value=str(gainers)
        )
    
    with col4:
        losers = len(stocks_df[stocks_df['change_percent'] < 0]) if 'change_percent' in stocks_df.columns else 0
        st.metric(
            label="Losers",
            value=str(losers)
        )
    
    # Most Active Stocks Table
    st.subheader("🔥 Most Active Stocks")
    
    # Format the display DataFrame
    display_df = stocks_df.copy()
    if 'change_percent' in display_df.columns:
        display_df['change_percent'] = display_df['change_percent'].apply(
            lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%" if not pd.isna(x) else "N/A"
        )
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )
    
    # Stock Selection and Analysis
    st.subheader("📈 Individual Stock Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if 'symbol' in stocks_df.columns and not stocks_df['symbol'].empty:
            selected_symbol = st.selectbox(
                "Select a stock for detailed analysis:",
                options=stocks_df['symbol'].tolist(),
                index=0
            )
        else:
            st.error("No stock symbols available")
            return
    
    with col2:
        period_options = {
            "1 Week": "1wk",
            "1 Month": "1mo",
            "3 Months": "3mo",
            "6 Months": "6mo",
            "1 Year": "1y"
        }
        selected_period = st.selectbox(
            "Select time period:",
            options=list(period_options.keys()),
            index=1
        )
    
    # Fetch and display stock data
    if selected_symbol:
        period_code = period_options[selected_period]
        
        with st.spinner(f"Loading {selected_symbol} data for {selected_period}..."):
            stock_data = data_fetcher.fetch_stock_history(selected_symbol, period_code)
        
        if stock_data is not None and not stock_data.empty:
            # Display metrics
            latest_data = stock_data.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Current Price",
                    value=f"${latest_data['close']:.2f}",
                    delta=f"{((latest_data['close'] - stock_data.iloc[-2]['close']) / stock_data.iloc[-2]['close'] * 100):.2f}%"
                )
            
            with col2:
                st.metric(
                    label="Volume",
                    value=f"{latest_data['volume']:,.0f}"
                )
            
            with col3:
                if 'rsi' in latest_data:
                    rsi_value = latest_data['rsi']
                    rsi_signal = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
                    st.metric(
                        label="RSI",
                        value=f"{rsi_value:.1f}",
                        help=f"Signal: {rsi_signal}"
                    )
            
            with col4:
                high_52_week = stock_data['high'].max()
                low_52_week = stock_data['low'].min()
                st.metric(
                    label="52W High",
                    value=f"${high_52_week:.2f}",
                    help=f"52W Low: ${low_52_week:.2f}"
                )
            
            # Display advanced chart
            fig = visualizer.create_advanced_chart(stock_data, selected_symbol)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display raw data
            with st.expander("📊 Raw Data"):
                st.dataframe(stock_data.tail(20))
        
        else:
            st.error(f"Could not fetch data for {selected_symbol}. Please try another symbol.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "📊 **Data provided by Yahoo Finance** | "
        "🔄 **Real-time updates available** | "
        f"🕒 **Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**"
    )

if __name__ == "__main__":
    main()