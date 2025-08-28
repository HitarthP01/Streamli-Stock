"""
Simple Modular Backup Application
A simplified version of the stock market dashboard with modular structure.
"""

import os
import pandas as pd
import streamlit as st
from datetime import datetime
import time
import logging
import sqlite3
import yfinance as yf
from bs4 import BeautifulSoup
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
CUSTOM_CSS = """
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
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

class DatabaseManager:
    def __init__(self, db_name: str = "backup_v1.db"):
        self.db_name = db_name
        self.init_database()
    
    def get_connection(self):
        return sqlite3.connect(self.db_name, check_same_thread=False)
    
    def init_database(self):
        """Initialize database with proper schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Most active stocks table
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
            
            # Stock prices table
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
            
            conn.commit()

class StockDataFetcher:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9" 
        }
    
    def fetch_most_active_stocks(self) -> Optional[pd.DataFrame]:
        """Fetch most active stocks from Yahoo Finance"""
        try:
            url = 'https://finance.yahoo.com/research-hub/screener/most_actives/'
            
            with st.spinner("Fetching most active stocks..."):
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                table = soup.find('table')
                
                if not table:
                    st.error("Could not find stock data table on the webpage")
                    return None
                
                # Extract headers
                header_cells = table.find('thead').find_all('th')
                columns = [h.get_text(strip=True) for h in header_cells] 
                
                # Extract data rows
                rows = table.find('tbody').find_all('tr')
                
                data = []
                for row in rows:
                    cells = [td.text for td in row.find_all('td')]
                    if len(cells) == len(columns):
                        data.append(cells)
                
                if not data:
                    st.warning("No stock data found")
                    return None
                
                df = pd.DataFrame(data, columns=columns)
                df['scrape_date'] = datetime.now().strftime('%Y-%m-%d')
                
                # Clean and process data
                df = self._clean_stock_data(df)
                
                # Save to database
                self._save_active_stocks_to_db(df)
                
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
        
        # Clean symbol column
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
        if not volume_str or pd.isna(volume_str):
            return 0
        s = str(volume_str).upper().replace(',', '').strip()
        try:
            if s.endswith('T'):
                return int(float(s[:-1]) * 1_000_000_000_000)
            elif s.endswith('B'):
                return int(float(s[:-1]) * 1_000_000_000)
            elif s.endswith('M'):
                return int(float(s[:-1]) * 1_000_000)
            elif s.endswith('K'):
                return int(float(s[:-1]) * 1_000)
            else:
                return int(float(s))
        except (ValueError, TypeError):
            return 0
    
    def _save_active_stocks_to_db(self, df: pd.DataFrame):
        """Save active stocks to database"""
        try:
            # Define required columns
            required_cols = [
                'symbol', 'name', 'price_intraday', 'change_amount', 'change_percent',
                'volume', 'avg_vol_3m', 'market_cap', 'pe_ratio', 'week_52_range', 'scrape_date'
            ]
            
            # Ensure all required columns exist
            for col in required_cols:
                if col not in df.columns:
                    df[col] = None
            
            # Select only required columns
            insert_df = df[required_cols].copy()
            
            # Fill NaN values appropriately
            string_cols = ['symbol', 'name', 'market_cap', 'week_52_range']
            numeric_cols = ['price_intraday', 'change_amount', 'change_percent', 'volume', 'avg_vol_3m', 'pe_ratio']
            
            for col in string_cols:
                if col in insert_df.columns:
                    insert_df[col] = insert_df[col].fillna('')
            
            for col in numeric_cols:
                if col in insert_df.columns:
                    insert_df[col] = insert_df[col].fillna(0.0)
            
            # Save to database
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Convert DataFrame to records
                records = insert_df.to_records(index=False)
                
                placeholders = ",".join(["?"] * len(required_cols))
                col_list = ",".join(required_cols)
                
                insert_sql = f"INSERT OR REPLACE INTO most_active_stocks ({col_list}) VALUES ({placeholders})"
                cursor.executemany(insert_sql, records)
                
                conn.commit()
                st.success(f"Saved {len(records)} stock records to database")
                
        except Exception as e:
            st.error(f"Error saving data to database: {str(e)}")
            logger.error(f"Database error: {e}")

class StockVisualizer:
    @staticmethod
    def create_simple_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
        """Create a simple price chart"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            )
        )
        
        fig.update_layout(
            title=f"{symbol.upper()} - Stock Price",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_white",
            height=400
        )
        
        return fig

def setup_sidebar():
    """Setup sidebar controls"""
    st.sidebar.title("⚙️ Dashboard Settings")
    
    # Manual refresh
    refresh_clicked = st.sidebar.button("🔄 Refresh Data Now")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "📊 Data Source:",
        ["Live Data (Yahoo)", "Database Cache"]
    )
    
    return data_source, refresh_clicked

def display_stock_table(stocks_df):
    """Display the most active stocks table"""
    st.subheader("🔥 Most Active Stocks")
    
    # Format the display DataFrame
    display_df = stocks_df.copy()
    if 'change_percent' in display_df.columns and not display_df.empty:
        display_df['change_percent'] = display_df['change_percent'].apply(
            lambda x: f"+{x:.2f}%" if pd.notna(x) and x > 0 else f"{x:.2f}%" if pd.notna(x) else "N/A"
        )
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )

def load_stock_data(data_source, db_manager, data_fetcher):
    """Load stock data based on selected source"""
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
        
        return stocks_df
        
    except Exception as e:
        st.error(f"Error loading stock data: {str(e)}")
        return None

def main():
    """Main application function"""
    try:
        # Initialize components
        db_manager = DatabaseManager()
        data_fetcher = StockDataFetcher(db_manager)
        visualizer = StockVisualizer()
        
        # Header
        st.markdown('<h1 class="main-header">📈 Stock Market Dashboard (Modular)</h1>', unsafe_allow_html=True)
        
        # Setup sidebar
        data_source, refresh_clicked = setup_sidebar()
        
        # Clear cache if refresh clicked
        if refresh_clicked:
            st.cache_data.clear()
            st.rerun()
        
        # Load stock data
        stocks_df = load_stock_data(data_source, db_manager, data_fetcher)
        
        if stocks_df is None or stocks_df.empty:
            st.error("No stock data available. Please try refreshing or check your connection.")
            return
        
        # Display most active stocks table
        display_stock_table(stocks_df)
        
        # Footer
        st.markdown("---")
        st.markdown(
            "📊 **Data provided by Yahoo Finance** | "
            f"🕒 **Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**"
        )
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Main application error: {e}")

if __name__ == "__main__":
    main()
