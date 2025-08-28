"""
This is a backup database for the stock market application.
Where we will implement a more robust data storage solution.


-------------------------------"""

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
import traceback


st.set_page_config(page_title="Active Stocks backup.py", page_icon="fire")

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




# class DatabaseManager:
    # def __init__(self, db_name: str = "backup_v1.db"):
    #     self.db_name = db_name
    #     self.init_database()
    
    # def get_connection(self):
    #     """Get database connection with timeout and retry logic"""
    #     import time
    #     max_retries = 5
    #     for attempt in range(max_retries):
    #         try:
    #             conn = sqlite3.connect(
    #                 self.db_name, 
    #                 check_same_thread=False,
    #                 timeout=30.0  # 30 second timeout
    #             )
    #             # Enable WAL mode for better concurrency
    #             conn.execute("PRAGMA journal_mode=WAL")
    #             conn.execute("PRAGMA synchronous=NORMAL")
    #             conn.execute("PRAGMA temp_store=memory")
    #             conn.execute("PRAGMA mmap_size=268435456")  # 256MB
    #             return conn
    #         except sqlite3.OperationalError as e:
    #             if "database is locked" in str(e).lower() and attempt < max_retries - 1:
    #                 logger.warning(f"Database locked, attempt {attempt + 1}/{max_retries}, retrying in {2 ** attempt} seconds...")
    #                 time.sleep(2 ** attempt)  # Exponential backoff
    #                 continue
    #             else:
    #                 raise e
        # raise sqlite3.OperationalError("Could not acquire database lock after multiple attempts")
class DatabaseManager:
    def __init__(self, db_name: str = "backup_v1.db"):
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
                is_active BOOLEAN DEFAULT TRUE
            )
            """)
            
            # Options chain table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS options_chain (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                contract_symbol TEXT NOT NULL,
                expiration_date DATE,
                strike REAL,
                option_type TEXT,
                last_price REAL,
                bid REAL,
                ask REAL,
                change_amount REAL,
                change_percent REAL,
                volume INTEGER,
                open_interest INTEGER,
                implied_volatility REAL,
                in_the_money BOOLEAN,
                contract_size INTEGER,
                currency TEXT,
                last_trade_date TIMESTAMP,
                fetch_date DATE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(contract_symbol, fetch_date)
            )
            """)
            
            # Create indexes for better performance
            # cursor.execute("CREATE INDEX IF NOT EXISTS idx_stock_symbol_date ON stock_prices(symbol, date)")
            # cursor.execute("CREATE INDEX IF NOT EXISTS idx_active_stocks_date ON most_active_stocks(scrape_date)")
            # cursor.execute("CREATE INDEX IF NOT EXISTS idx_active_stocks_symbol ON most_active_stocks(symbol)")
            # cursor.execute("CREATE INDEX IF NOT EXISTS idx_options_symbol ON options_chain(symbol)")
            # cursor.execute("CREATE INDEX IF NOT EXISTS idx_options_volume ON options_chain(volume)")
            
            conn.commit()

class StockDataFetcher:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9" 
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
                # print(df.columns.tolist())
                
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
    
                
    # def fetch_option_chain(self, symbol: str, min_volume: int = 400) -> Optional[pd.DataFrame]:
    #     """Fetch options chain data for a given symbol with volume filter"""
    #     try:
    #         with st.spinner(f"Fetching options data for {symbol}..."):
    #             stock = yf.Ticker(symbol)
    #             options = stock.options
                
    #             if not options:
    #                 st.warning(f"No options available for {symbol}")
    #                 return None

    #             all_options_data = []
                
    #             # Fetch options for all expiration dates (limit to first 3 for performance)
    #             for exp_date in options[:3]:  # Limit to first 3 expiration dates
    #                 try:
    #                     option_chain = stock.option_chain(exp_date)
    #                     calls = option_chain.calls.copy()
    #                     puts = option_chain.puts.copy()

    #                     # Add metadata
    #                     calls['type'] = 'call'
    #                     calls['expiration_date'] = exp_date
    #                     calls['symbol'] = symbol.upper()
                        
    #                     puts['type'] = 'put'
    #                     puts['expiration_date'] = exp_date
    #                     puts['symbol'] = symbol.upper()

    #                     # Combine calls and puts
    #                     exp_options = pd.concat([calls, puts], ignore_index=True)
                        
    #                     # Filter by volume
    #                     if min_volume > 0:
    #                         exp_options = exp_options[exp_options['volume'] >= min_volume]
                        
    #                     all_options_data.append(exp_options)
                        
    #                 except Exception as e:
    #                     logger.warning(f"Error fetching options for {symbol} expiration {exp_date}: {e}")
    #                     continue
                
    #             if not all_options_data:
    #                 st.warning(f"No options data with volume >= {min_volume} found for {symbol}")
    #                 return None
                
    #             # Combine all expiration dates
    #             option_data = pd.concat(all_options_data, ignore_index=True)
                
    #             if option_data.empty:
    #                 st.warning(f"No options data with volume >= {min_volume} found for {symbol}")
    #                 return None
                
    #             # Clean and standardize the data
    #             option_data = self._clean_options_data(option_data)
                
    #             # Save to database
    #             self._save_options_to_db(option_data)
                
    #             return option_data
                
    #     except Exception as e:
    #         st.error(f"Error fetching options chain for {symbol}: {str(e)}")
    #         logger.error(f"Error fetching options chain for {symbol}: {e}")
    #         return None
    
    def _clean_options_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize options data"""
        if df.empty:
            return df
            
        # Standardize column names
        column_mapping = {
            'contractSymbol': 'contract_symbol',
            'lastTradeDate': 'last_trade_date',
            'lastPrice': 'last_price',
            'percentChange': 'change_percent',
            'openInterest': 'open_interest',
            'impliedVolatility': 'implied_volatility',
            'inTheMoney': 'in_the_money',
            'contractSize': 'contract_size',
            'change': 'change_amount',
            'type': 'option_type'
        }
        
        # Rename columns that exist
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Add fetch date
        df['fetch_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Convert data types
        numeric_columns = ['strike', 'last_price', 'bid', 'ask', 'change_amount', 'change_percent', 
                          'implied_volatility', 'current_stock_price', 'strike_vs_price_pct']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        integer_columns = ['volume', 'open_interest', 'contract_size']
        for col in integer_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # Convert boolean columns
        if 'in_the_money' in df.columns:
            df['in_the_money'] = df['in_the_money'].astype(bool)
        
        # Convert timestamps
        if 'last_trade_date' in df.columns:
            df['last_trade_date'] = pd.to_datetime(df['last_trade_date'], errors='coerce')
        
        return df

    def fetch_option_chain(self, symbol: str, min_volume: int = 400, weeks_ahead: int = 10, 
                        price_range_percent: float = 10.0) -> Optional[pd.DataFrame]:
        """
        Fetch options chain data with enhanced filtering:
        - Strike prices within ±10% of current stock price
        - Options expiring within next 10 weeks
        - Volume > 400 for both calls and puts
        """
        try:
            with st.spinner(f"Fetching options data for {symbol}..."):
                stock = yf.Ticker(symbol)
                
                # Step 1: Get current stock price
                try:
                    # Method 1: Try from stock.info
                    current_price = stock.info.get('regularMarketPrice')
                    if not current_price or pd.isna(current_price):
                        # Method 2: Fallback to recent history
                        hist = stock.history(period='1d', interval='1m')
                        if not hist.empty:
                            current_price = hist['Close'].iloc[-1]
                        else:
                            # Method 3: Last resort - 5 day history
                            hist = stock.history(period='5d')
                            current_price = hist['Close'].iloc[-1] if not hist.empty else None
                            
                    if not current_price:
                        st.error(f"Could not get current price for {symbol}")
                        return None
                        
                except Exception as e:
                    st.error(f"Error getting current price for {symbol}: {e}")
                    return None
                
                # Step 2: Calculate ±10% price range
                price_range_decimal = price_range_percent / 100.0
                min_strike = current_price * (1 - price_range_decimal)  # 90% of current price
                max_strike = current_price * (1 + price_range_decimal)  # 110% of current price
                
                st.info(f"Current price: ${current_price:.2f} | Strike range: ${min_strike:.2f} - ${max_strike:.2f}")
                
                # Step 3: Get all available expiration dates
                options = stock.options
                if not options:
                    st.warning(f"No options available for {symbol}")
                    return None

                # Step 4: Filter expiration dates within next 10 weeks
                current_date = datetime.now().date()
                target_date = current_date + timedelta(weeks=weeks_ahead)
                
                valid_expirations = []
                for exp_str in options:
                    exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                    if current_date <= exp_date <= target_date:
                        valid_expirations.append(exp_str)
                
                if not valid_expirations:
                    st.warning(f"No options expiring within {weeks_ahead} weeks for {symbol}")
                    return None
                    
                st.info(f"Found {len(valid_expirations)} expiration dates within {weeks_ahead} weeks")

                all_options_data = []
                
                # Step 5: Process each valid expiration date
                for exp_date in valid_expirations:
                    try:
                        option_chain = stock.option_chain(exp_date)
                        calls = option_chain.calls.copy()
                        puts = option_chain.puts.copy()

                        # Step 6: Filter by strike price range (±10%)
                        calls_filtered = calls[
                            (calls['strike'] >= min_strike) & 
                            (calls['strike'] <= max_strike)
                        ].copy()
                        
                        puts_filtered = puts[
                            (puts['strike'] >= min_strike) & 
                            (puts['strike'] <= max_strike)
                        ].copy()

                        # Step 7: Filter by volume (separate for calls and puts)
                        calls_filtered = calls_filtered[calls_filtered['volume'] >= min_volume]
                        puts_filtered = puts_filtered[puts_filtered['volume'] >= min_volume]

                        # Add metadata
                        if not calls_filtered.empty:
                            calls_filtered['type'] = 'call'
                            calls_filtered['expiration_date'] = exp_date
                            calls_filtered['symbol'] = symbol.upper()
                            calls_filtered['current_stock_price'] = current_price
                            # Calculate how close strike is to current price
                            calls_filtered['strike_vs_price_pct'] = (
                                (calls_filtered['strike'] - current_price) / current_price * 100
                            )
                            
                        if not puts_filtered.empty:
                            puts_filtered['type'] = 'put'
                            puts_filtered['expiration_date'] = exp_date
                            puts_filtered['symbol'] = symbol.upper()
                            puts_filtered['current_stock_price'] = current_price
                            puts_filtered['strike_vs_price_pct'] = (
                                (puts_filtered['strike'] - current_price) / current_price * 100
                            )

                        # Combine calls and puts for this expiration
                        if not calls_filtered.empty or not puts_filtered.empty:
                            exp_options = pd.concat([calls_filtered, puts_filtered], ignore_index=True)
                            all_options_data.append(exp_options)
                            
                            # Log summary for this expiration
                            calls_count = len(calls_filtered)
                            puts_count = len(puts_filtered)
                            st.success(f"Expiration {exp_date}: {calls_count} calls, {puts_count} puts (volume ≥ {min_volume})")
                            
                    except Exception as e:
                        logger.warning(f"Error processing {symbol} expiration {exp_date}: {e}")
                        continue
                
                if not all_options_data:
                    st.warning(f"No options found matching criteria for {symbol}")
                    return None
                
                # Step 8: Combine all expiration dates
                option_data = pd.concat(all_options_data, ignore_index=True)
                
                if option_data.empty:
                    st.warning(f"No options data found matching all criteria for {symbol}")
                    return None
                
                # Step 9: Sort by expiration date and volume (descending)
                option_data = option_data.sort_values(['expiration_date', 'volume'], ascending=[True, False])
                
                # Display summary
                total_calls = len(option_data[option_data['type'] == 'call'])
                total_puts = len(option_data[option_data['type'] == 'put'])
                total_volume = option_data['volume'].sum()
                
                st.success(f"""
                📊 **Options Summary for {symbol}**
                - Current Stock Price: ${current_price:.2f}
                - Strike Range: ${min_strike:.2f} - ${max_strike:.2f} (±{price_range_percent}%)
                - Time Frame: Next {weeks_ahead} weeks
                - Total Options Found: {len(option_data)} ({total_calls} calls, {total_puts} puts)
                - Total Volume: {total_volume:,}
                - Min Volume Filter: {min_volume}
                """)
                
                # Clean and standardize the data
                option_data = self._clean_options_data(option_data)
                
                # Save to database
                self._save_options_to_db(option_data)
                
                return option_data
                    
        except Exception as e:
            st.error(f"Error fetching options chain for {symbol}: {str(e)}")
            logger.error(f"Error fetching options chain for {symbol}: {e}")
            return None

    def _save_options_to_db(self, df: pd.DataFrame):
        """Save options data to individual symbol-specific tables"""
        if df.empty:
            return
            
        conn = None
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            # Group by symbol to create separate tables for each
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol].copy()
                
                # Create table name (sanitize symbol name)
                clean_symbol = str(symbol).upper().replace('.', '_').replace('-', '_')
                clean_symbol = ''.join(c for c in clean_symbol if c.isalnum() or c == '_')
                table_name = f"options_{clean_symbol}"
                
                # Create table if it doesn't exist
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    contract_symbol TEXT NOT NULL,
                    expiration_date TEXT,
                    strike REAL,
                    option_type TEXT,
                    last_price REAL,
                    bid REAL,
                    ask REAL,
                    change_amount REAL,
                    change_percent REAL,
                    volume INTEGER,
                    open_interest INTEGER,
                    implied_volatility REAL,
                    in_the_money INTEGER,
                    contract_size INTEGER,
                    currency TEXT,
                    last_trade_date TEXT,
                    current_stock_price REAL,
                    strike_vs_price_pct REAL,
                    fetch_date TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(contract_symbol, fetch_date)
                )
                """)
                
                # Migration: Add new columns if they don't exist (for existing tables)
                try:
                    # Check if current_stock_price column exists
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = [column[1] for column in cursor.fetchall()]
                    
                    if 'current_stock_price' not in columns:
                        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN current_stock_price REAL")
                        logger.info(f"Added current_stock_price column to {table_name}")
                    
                    if 'strike_vs_price_pct' not in columns:
                        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN strike_vs_price_pct REAL")
                        logger.info(f"Added strike_vs_price_pct column to {table_name}")
                        
                except Exception as migration_error:
                    logger.warning(f"Migration warning for {table_name}: {migration_error}")
                    # Continue execution even if migration fails
                
                # Prepare data for this symbol
                expected_cols = [
                    'contract_symbol', 'expiration_date', 'strike', 'option_type',
                    'last_price', 'bid', 'ask', 'change_amount', 'change_percent',
                    'volume', 'open_interest', 'implied_volatility', 'in_the_money',
                    'contract_size', 'currency', 'last_trade_date', 'current_stock_price',
                    'strike_vs_price_pct', 'fetch_date'
                ]
                
                # Ensure all expected columns exist with proper data types
                for col in expected_cols:
                    if col not in symbol_df.columns:
                        if col in ['contract_symbol', 'expiration_date', 'option_type', 'currency', 'last_trade_date', 'fetch_date']:
                            symbol_df[col] = ''
                        elif col == 'in_the_money':
                            symbol_df[col] = 0  # Use 0/1 instead of boolean for SQLite
                        else:
                            symbol_df[col] = 0.0
                
                # Select and clean data
                insert_df = symbol_df[expected_cols].copy()
                
                # Handle data type conversions for SQLite compatibility
                string_cols = ['contract_symbol', 'expiration_date', 'option_type', 'currency', 'last_trade_date', 'fetch_date']
                for col in string_cols:
                    if col in insert_df.columns:
                        insert_df[col] = insert_df[col].astype(str).fillna('')
                
                numeric_cols = ['strike', 'last_price', 'bid', 'ask', 'change_amount', 'change_percent', 
                              'implied_volatility', 'current_stock_price', 'strike_vs_price_pct']
                for col in numeric_cols:
                    if col in insert_df.columns:
                        insert_df[col] = pd.to_numeric(insert_df[col], errors='coerce').fillna(0.0)
                
                integer_cols = ['volume', 'open_interest', 'contract_size', 'in_the_money']
                for col in integer_cols:
                    if col in insert_df.columns:
                        insert_df[col] = pd.to_numeric(insert_df[col], errors='coerce').fillna(0).astype(int)
                
                # Convert boolean to integer for in_the_money
                if 'in_the_money' in insert_df.columns:
                    insert_df['in_the_money'] = insert_df['in_the_money'].astype(bool).astype(int)
                
                # Convert datetime to string for last_trade_date
                if 'last_trade_date' in insert_df.columns:
                    insert_df['last_trade_date'] = pd.to_datetime(insert_df['last_trade_date'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
                
                # Convert to records
                rows = insert_df.to_records(index=False)
                
                if len(rows) == 0:
                    continue
                
                # Insert with conflict resolution
                placeholders = ",".join(["?"] * len(expected_cols))
                col_list = ",".join(expected_cols)
                
                insert_sql = f"INSERT OR REPLACE INTO {table_name} ({col_list}) VALUES ({placeholders})"
                
                # Convert numpy types to Python types for SQLite compatibility
                converted_rows = []
                for row in rows:
                    converted_row = tuple(
                        item.item() if hasattr(item, 'item') else item 
                        for item in row
                    )
                    converted_rows.append(converted_row)
                
                try:
                    cursor.executemany(insert_sql, converted_rows)
                    logger.info(f"Saved {len(converted_rows)} options records to {table_name}")
                except Exception as insert_error:
                    # If insert fails, try with only the columns that exist in the table
                    logger.warning(f"Insert failed for {table_name}, trying fallback method: {insert_error}")
                    
                    # Get actual table columns
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    actual_columns = [column[1] for column in cursor.fetchall()]
                    
                    # Filter to only columns that exist in the table
                    available_cols = [col for col in expected_cols if col in actual_columns]
                    fallback_df = symbol_df[available_cols].copy()
                    
                    # Reprocess with available columns only
                    for col in ['contract_symbol', 'expiration_date', 'option_type', 'currency', 'last_trade_date', 'fetch_date']:
                        if col in fallback_df.columns:
                            fallback_df[col] = fallback_df[col].astype(str).fillna('')
                    
                    for col in ['strike', 'last_price', 'bid', 'ask', 'change_amount', 'change_percent', 'implied_volatility']:
                        if col in fallback_df.columns:
                            fallback_df[col] = pd.to_numeric(fallback_df[col], errors='coerce').fillna(0.0)
                    
                    for col in ['volume', 'open_interest', 'contract_size', 'in_the_money']:
                        if col in fallback_df.columns:
                            fallback_df[col] = pd.to_numeric(fallback_df[col], errors='coerce').fillna(0).astype(int)
                    
                    # Try insert with available columns
                    fallback_rows = fallback_df.to_records(index=False)
                    fallback_converted = [tuple(item.item() if hasattr(item, 'item') else item for item in row) for row in fallback_rows]
                    
                    fallback_placeholders = ",".join(["?"] * len(available_cols))
                    fallback_col_list = ",".join(available_cols)
                    fallback_sql = f"INSERT OR REPLACE INTO {table_name} ({fallback_col_list}) VALUES ({fallback_placeholders})"
                    
                    cursor.executemany(fallback_sql, fallback_converted)
                    logger.info(f"Saved {len(fallback_converted)} options records to {table_name} (fallback mode)")
                    st.warning(f"⚠️ Some new columns not available in {table_name}. Data saved with available columns.")
            
            conn.commit()
            logger.info(f"Successfully saved options data to individual symbol tables")
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error saving options data: {e}")
            traceback.print_exc()
        finally:
            if conn:
                conn.close()

    def get_options_from_db(self, symbol: str, min_volume: int = 1000) -> Optional[pd.DataFrame]:
        """Retrieve options data from symbol-specific table"""
        try:
            conn = self.db_manager.get_connection()
            today = datetime.now().strftime('%Y-%m-%d')
            
            # Create table name (sanitize symbol name)
            clean_symbol = str(symbol).lower().replace('.', '_').replace('-', '_')
            clean_symbol = ''.join(c for c in clean_symbol if c.isalnum() or c == '_')
            table_name = f"options_{clean_symbol}"
            
            # Check if table exists
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (table_name,))
            
            if not cursor.fetchone():
                logger.info(f"No options table found for {symbol}")
                conn.close()
                return None
            
            query = f"""
            SELECT * FROM {table_name}
            WHERE volume >= ? AND fetch_date = ?
            ORDER BY expiration_date, option_type, strike
            """
            
            df = pd.read_sql_query(query, conn, params=(min_volume, today))
            conn.close()
            
            if not df.empty:
                # Convert back from integer to boolean for in_the_money
                if 'in_the_money' in df.columns:
                    df['in_the_money'] = df['in_the_money'].astype(bool)
                
                # Add symbol column back
                df['symbol'] = symbol.upper()
            
            return df if not df.empty else None
            
        except Exception as e:
            logger.error(f"Error fetching options from database for {symbol}: {e}")
            return None

    def _parse_volume(self, market_cap_str: str) -> int:
        """Parse volume strings like '1.2M', '500K', etc."""
        if not market_cap_str:
            return 0
        s = market_cap_str.upper().replace(',', '').strip()
        try:
            if s.endswith('T'):
                return float(s[:-1]) * 1_000_000_000_000
            elif s.endswith('B'):
                return float(s[:-1]) * 1_000_000_000
            elif s.endswith('M'):
                return float(s[:-1]) * 1_000_000
            elif s.endswith('K'):
                return float(s[:-1]) * 1_000
            else:  # assume plain number
                return float(s)
        except:
            return 0  # fallback if string is invalid
            

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
        """Save active stocks to the database and create tables for stocks with market cap > 100B."""
        conn = None
        try:
            df = self._sanitize_columns(df)
            
            # Ensure scrape_date column exists
            if 'scrape_date' not in df.columns:
                df['scrape_date'] = datetime.now().strftime('%Y-%m-%d')

            # Filter stocks with market cap > 100 billion
            df['market_cap_numeric'] = df['market_cap'].apply(self._parse_volume)
            high_cap_stocks = df[df['market_cap_numeric'] > 100_000_000_000]
            
            cols = [
                'symbol', 'name', 'price_intraday', 'change_amount', 'change_percent',
                'volume', 'avg_vol_3m', 'market_cap', 'pe_ratio', 'week_52_range', 'scrape_date'
            ]
            
            # Ensure all required columns exist with appropriate defaults
            for col in cols:
                if col not in df.columns:
                    if col in ['symbol', 'name', 'market_cap', 'week_52_range']:
                        df[col] = None  # String columns get None
                    else:
                        df[col] = 0.0   # Numeric columns get 0.0

            # Select only the columns we want to insert
            insert_df = df[cols].copy()
            
            # Handle NaN values appropriately for each column type
            string_cols = ['symbol', 'name', 'market_cap', 'week_52_range']
            numeric_cols = ['price_intraday', 'change_amount', 'change_percent', 'volume', 'avg_vol_3m', 'pe_ratio']
            
            for col in string_cols:
                if col in insert_df.columns:
                    insert_df[col] = insert_df[col].fillna('')
            
            for col in numeric_cols:
                if col in insert_df.columns:
                    insert_df[col] = insert_df[col].fillna(0.0)
            
            # Convert to records for database insertion
            rows = insert_df.to_records(index=False)
            
            print(f"Inserting {len(rows)} rows into most_active_stocks")

            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            # Use INSERT OR IGNORE first, then UPDATE for existing records on same date
            placeholders = ",".join(["?"] * len(cols))
            col_list = ",".join(cols)
            
            # Insert new records (will ignore duplicates based on UNIQUE constraint)
            insert_sql = f"INSERT OR IGNORE INTO most_active_stocks ({col_list}) VALUES ({placeholders})"
            cursor.executemany(insert_sql, rows)
            inserted_count = cursor.rowcount
            
            # Update existing records for the same date (if running multiple times same day)
            update_cols = [col for col in cols if col not in ['symbol', 'scrape_date']]  # Don't update key columns
            update_set = ", ".join([f"{col} = ?" for col in update_cols])
            
            update_sql = f"""
            UPDATE most_active_stocks 
            SET {update_set}
            WHERE symbol = ? AND scrape_date = ?
            """
            updated_count = 0
            for row in rows:
                row_dict = dict(zip(cols, row))
                update_values = [row_dict[col] for col in update_cols]  # Values for SET clause
                update_values.extend([row_dict['symbol'], row_dict['scrape_date']])  # Values for WHERE clause
                
                cursor.execute(update_sql, update_values)
                if cursor.rowcount > 0:
                    updated_count += 1
            
            print(f"Inserted {inserted_count} new rows, updated {updated_count} existing rows")

            MINIMUM_HISTORICAL_DAYS = 60  # Minimum days needed for reliable indicators
            
            # Create tables for high-cap stocks with proper OHLCV data
            if not high_cap_stocks.empty:
                print(f"Processing {len(high_cap_stocks)} high-cap stocks for individual tables")
                
                for _, row in high_cap_stocks.iterrows():
                    symbol = str(row['symbol']).lower().replace('.', '_').replace('-', '_')
                    symbol = ''.join(c for c in symbol if c.isalnum() or c == '_')
                    table_name = f"{symbol}"
                    
                    # Create table if it doesn't exist
                    cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                        UNIQUE(date)
                    )
                    """)

                    # Check current data status
                    data_status = self._assess_data_completeness(cursor, table_name, row['scrape_date'])
                    print(f"Data status is : {data_status['action']}")

                    if data_status['action'] == 'fetch_full_history':
                        print(f"Insufficient data for {row['symbol']} ({data_status['record_count']} days) - fetching full historical data")
                        self._fetch_and_store_historical_data(cursor, table_name, row['symbol'])
                        
                    elif data_status['action'] == 'add_single_day':
                        print(f"Adding new day's data for {row['symbol']}")
                        self._fetch_and_store_single_day(cursor, table_name, row['symbol'], row['scrape_date'])
                        
                    elif data_status['action'] == 'already_exists':
                        print(f"Data already exists for {row['symbol']} on {row['scrape_date']}")
                        
                    elif data_status['action'] == 'backfill_and_add':
                        print(f"Backfilling missing data for {row['symbol']} and adding today's data")
                        self._backfill_historical_data(cursor, table_name, row['symbol'])
                        self._fetch_and_store_single_day(cursor, table_name, row['symbol'], row['scrape_date'])

            conn.commit()
            print("Successfully committed all changes to database")
        
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error saving active stocks to database: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if conn:
                conn.close()
    def _fetch_and_store_historical_data(self, cursor, table_name: str, symbol: str, days: int = 60):
        """Fetch and store historical data for a new stock."""
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            
            # Get historical data for the specified number of days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            hist = ticker.history(
                start=start_date.strftime('%Y-%m-%d'), 
                end=end_date.strftime('%Y-%m-%d')
            )
            
            if hist.empty:
                print(f"No historical data available for {symbol}")
                return
            
            # Convert to DataFrame for technical indicators
            df = hist.reset_index()
            df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
            df['close'] = df['Close']
            
            # Calculate technical indicators using your existing function
            df = self._calculate_technical_indicators(df)
            
            # Prepare data for insertion
            records_inserted = 0
            for _, row in df.iterrows():
                stock_data = (
                    row['date'],
                    round(float(row['Open']), 3),
                    round(float(row['High']), 3),
                    round(float(row['Low']), 3),
                    round(float(row['Close']), 3),
                    round(float(row['Close']), 3),  # adj_close
                    int(row['Volume']),
                    round(float(row['sma_20']), 3) if pd.notna(row['sma_20']) else None,
                    round(float(row['sma_50']), 3) if pd.notna(row['sma_50']) else None,
                    round(float(row['rsi']), 3) if pd.notna(row['rsi']) else None
                )
                
                try:
                    insert_sql = f"""
                    INSERT OR IGNORE INTO {table_name} 
                    (date, open, high, low, close, adj_close, volume, sma_20, sma_50, rsi) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    cursor.execute(insert_sql, stock_data)
                    if cursor.rowcount > 0:
                        records_inserted += 1
                        
                except Exception as e:
                    logger.warning(f"Error inserting data for {symbol} on {row['date']}: {e}")
                    continue
            
            print(f"Inserted {records_inserted} historical records for {symbol}")
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")

    def _fetch_and_store_single_day(self, cursor, table_name: str, symbol: str, date: str):
        """Fetch and store data for a single day, calculating indicators based on existing data."""
        try:
            # First, get the OHLCV data for the specific date
            ohlcv_data = self._fetch_ohlcv_data(symbol, date)
            
            if not ohlcv_data:
                print(f"No OHLCV data available for {symbol} on {date}")
                return
            
            # Insert the basic OHLCV data first
            stock_data_base = (
                date,
                round(ohlcv_data['open'], 3),
                round(ohlcv_data['high'], 3),
                round(ohlcv_data['low'], 3),
                round(ohlcv_data['close'], 3),
                round(ohlcv_data['adj_close'], 3),
                ohlcv_data['volume'],
                None,  # Will calculate indicators next
                None,
                None
            )
            
            insert_sql = f"""
            INSERT INTO {table_name} 
            (date, open, high, low, close, adj_close, volume, sma_20, sma_50, rsi) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(insert_sql, stock_data_base)
            
            # Now calculate and update technical indicators for this date
            self._update_single_day_indicators(cursor, table_name, symbol, date)
            
            print(f"Added data for {symbol} on {date}")
            
        except Exception as e:
            logger.error(f"Error adding single day data for {symbol}: {e}")

    def _update_single_day_indicators(self, cursor, table_name: str, symbol: str, target_date: str):
        """Calculate technical indicators for a specific date using historical data in DB."""
        try:
            # Get all historical data up to and including the target date
            cursor.execute(f"""
                SELECT date, close FROM {table_name} 
                WHERE date <= ? AND close IS NOT NULL 
                ORDER BY date ASC
            """, (target_date,))
            
            data = cursor.fetchall()
            
            if len(data) < 1:
                print(f"No historical data available for indicators calculation for {symbol}")
                return
            
            # Create DataFrame and calculate indicators
            df = pd.DataFrame(data, columns=['date', 'close'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Calculate technical indicators
            df = self._calculate_technical_indicators(df)
            
            # Get indicators for the target date
            target_row = df[df['date'] == pd.to_datetime(target_date)]
            
            if not target_row.empty:
                target_row = target_row.iloc[-1]
                
                # Update the database with calculated indicators
                cursor.execute(f"""
                    UPDATE {table_name} 
                    SET sma_20 = ?, sma_50 = ?, rsi = ?
                    WHERE date = ?
                """, (
                    round(float(target_row['sma_20']), 3) if pd.notna(target_row['sma_20']) else None,
                    round(float(target_row['sma_50']), 3) if pd.notna(target_row['sma_50']) else None,
                    round(float(target_row['rsi']), 3) if pd.notna(target_row['rsi']) else None,
                    target_date
                ))
                
                print(f"Updated indicators for {symbol} on {target_date}")
            
        except Exception as e:
            logger.error(f"Error updating indicators for {symbol} on {target_date}: {e}")

    def _assess_data_completeness(self, cursor, table_name: str, target_date: str, min_days: int = 60):
        """Assess the completeness of data in the stock table and determine action needed."""
        
        # Check total records
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_records = cursor.fetchone()[0]
        
        # Check if today's data already exists
        cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE date = ?", (target_date,))
        today_exists = cursor.fetchone()[0] > 0
        
        # Get date range of existing data
        cursor.execute(f"""
            SELECT MIN(date) as earliest, MAX(date) as latest, COUNT(*) as count
            FROM {table_name}
        """)
        result = cursor.fetchone()
        earliest_date, latest_date, record_count = result
        
        # Calculate days between earliest and latest
        if earliest_date and latest_date:
            earliest = datetime.strptime(earliest_date, '%Y-%m-%d')
            latest = datetime.strptime(latest_date, '%Y-%m-%d')
            date_span_days = (latest - earliest).days + 1
            
            # Calculate data density (how complete is the data in the span)
            data_density = record_count / date_span_days if date_span_days > 0 else 0
        else:
            date_span_days = 0
            data_density = 0
        
        # Decision logic
        if total_records == 0:
            action = 'fetch_full_history'
        elif today_exists:
            action = 'already_exists'
        elif total_records < min_days:
            # Not enough historical data for reliable indicators
            action = 'fetch_full_history'
        elif data_density < 0.7:  # Less than 70% data completeness
            # Has some data but too many gaps
            action = 'backfill_and_add'
        else:
            # Good data, just add today
            action = 'add_single_day'
        
        return {
            'action': action,
            'record_count': record_count,
            'earliest_date': earliest_date,
            'latest_date': latest_date,
            'date_span_days': date_span_days,
            'data_density': data_density
        }

    def _backfill_historical_data(self, cursor, table_name: str, symbol: str, days_back: int = 60):
        """Backfill missing historical data for existing table."""
        try:
            # Get the earliest date in the table
            cursor.execute(f"SELECT MIN(date) FROM {table_name}")
            earliest_date_str = cursor.fetchone()[0]
            
            if not earliest_date_str:
                # No data at all, fetch full history
                self._fetch_and_store_historical_data(cursor, table_name, symbol, days_back)
                return
            
            earliest_date = datetime.strptime(earliest_date_str, '%Y-%m-%d')
            target_start_date = earliest_date - timedelta(days=days_back)
            
            print(f"Backfilling {symbol} from {target_start_date.strftime('%Y-%m-%d')} to {earliest_date_str}")
            
            # Fetch historical data before the earliest date
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            
            hist = ticker.history(
                start=target_start_date.strftime('%Y-%m-%d'),
                end=earliest_date.strftime('%Y-%m-%d')  # Up to but not including earliest
            )
            
            if hist.empty:
                print(f"No backfill data available for {symbol}")
                return
            
            # Process and insert backfill data
            records_inserted = 0
            for date, row in hist.iterrows():
                date_str = date.strftime('%Y-%m-%d')
                
                stock_data = (
                    date_str,
                    round(float(row['Open']), 3),
                    round(float(row['High']), 3),
                    round(float(row['Low']), 3),
                    round(float(row['Close']), 3),
                    round(float(row['Close']), 3),
                    int(row['Volume']),
                    None, None, None  # Will calculate indicators after all backfill is done
                )
                
                try:
                    insert_sql = f"""
                    INSERT OR IGNORE INTO {table_name} 
                    (date, open, high, low, close, adj_close, volume, sma_20, sma_50, rsi) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    cursor.execute(insert_sql, stock_data)
                    if cursor.rowcount > 0:
                        records_inserted += 1
                        
                except Exception as e:
                    logger.warning(f"Error inserting backfill data for {symbol} on {date_str}: {e}")
            
            print(f"Backfilled {records_inserted} records for {symbol}")
            
            # Now recalculate all technical indicators for the entire dataset
            self._recalculate_all_indicators(cursor, table_name, symbol)
            
        except Exception as e:
            logger.error(f"Error backfilling data for {symbol}: {e}")

    def _recalculate_all_indicators(self, cursor, table_name: str, symbol: str):
        """Recalculate technical indicators for all data in the table."""
        try:
            # Get all data ordered by date
            cursor.execute(f"""
                SELECT date, close FROM {table_name} 
                WHERE close IS NOT NULL 
                ORDER BY date ASC
            """)
            
            data = cursor.fetchall()
            if len(data) < 1:
                return
            
            # Create DataFrame and calculate indicators
            df = pd.DataFrame(data, columns=['date', 'close'])
            df['date'] = pd.to_datetime(df['date'])
            df = self._calculate_technical_indicators(df)
            
            # Update all records with new indicators
            updated_count = 0
            for _, row in df.iterrows():
                cursor.execute(f"""
                    UPDATE {table_name} 
                    SET sma_20 = ?, sma_50 = ?, rsi = ?
                    WHERE date = ?
                """, (
                    round(float(row['sma_20']), 3) if pd.notna(row['sma_20']) else None,
                    round(float(row['sma_50']), 3) if pd.notna(row['sma_50']) else None,
                    round(float(row['rsi']), 3) if pd.notna(row['rsi']) else None,
                    row['date'].strftime('%Y-%m-%d')
                ))
                updated_count += 1
            
            print(f"Recalculated indicators for {updated_count} records in {symbol}")
            
        except Exception as e:
            logger.error(f"Error recalculating indicators for {symbol}: {e}")

    def _fetch_ohlcv_data(self, symbol: str, date: str):
        """Fetch OHLCV data using yfinance."""
        try:            
            ticker = yf.Ticker(symbol)
            end_date = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)
            start_date = datetime.strptime(date, '%Y-%m-%d')
            
            hist = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                                end=end_date.strftime('%Y-%m-%d'))
            
            if not hist.empty:
                row = hist.iloc[0]
                return {
                    'open': round(float(row['Open']), 3),
                    'high': round(float(row['High']), 3),
                    'low': round(float(row['Low']), 3),
                    'close': round(float(row['Close']), 3),
                    'adj_close': round(float(row['Close']), 3),
                    'volume': int(row['Volume'])
                }
            else:
                print(f"No OHLCV data found for {symbol} on {date}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return None

    def _fetch_ohlcv_data_with_retry(self, symbol: str, date: str, max_retries: int = 3):
        """Fetch OHLCV data with retry logic for better reliability."""
        for attempt in range(max_retries):
            try:
                data = self._fetch_ohlcv_data(symbol, date)
                if data:
                    return data
                else:
                    logger.warning(f"No OHLCV data for {symbol} on {date}, attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Wait 1 second before retry
            except Exception as e:
                logger.warning(f"Error fetching OHLCV for {symbol} on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        logger.error(f"Failed to fetch OHLCV data for {symbol} on {date} after {max_retries} attempts")
        return None

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
        """Save price data to database using robust INSERT OR IGNORE logic"""
        conn = None
        try:
            if df.empty:
                logger.info("No price data to save - DataFrame is empty")
                return
                
            # Sanitize column names
            df = df.copy()
            df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
            
            # Define expected columns for stock_prices table
            expected_cols = [
                'symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 
                'volume', 'sma_20', 'sma_50', 'rsi'
            ]
            
            # Ensure all expected columns exist with appropriate defaults
            for col in expected_cols:
                if col not in df.columns:
                    if col in ['symbol', 'date']:
                        df[col] = None  # String columns get None
                    else:
                        df[col] = 0.0   # Numeric columns get 0.0
            
            # Select only the columns we want to insert
            insert_df = df[expected_cols].copy()
            
            # Handle date formatting
            if 'date' in insert_df.columns:
                insert_df['date'] = pd.to_datetime(insert_df['date']).dt.strftime('%Y-%m-%d')
            
            # Handle NaN values appropriately for each column type
            string_cols = ['symbol', 'date']
            numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume', 'sma_20', 'sma_50', 'rsi']
            
            for col in string_cols:
                if col in insert_df.columns:
                    insert_df[col] = insert_df[col].fillna('')
            
            for col in numeric_cols:
                if col in insert_df.columns:
                    insert_df[col] = insert_df[col].fillna(0.0)
            
            # Convert to records for database insertion
            rows = insert_df.to_records(index=False)
            
            if len(rows) == 0:
                logger.info("No price rows to save after processing")
                return
                
            print(f"Inserting {len(rows)} rows into stock_prices")
            
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            # Use INSERT OR IGNORE first, then UPDATE for existing records
            placeholders = ",".join(["?"] * len(expected_cols))
            col_list = ",".join(expected_cols)
            
            # Insert new records (will ignore duplicates based on UNIQUE constraint)
            insert_sql = f"INSERT OR IGNORE INTO stock_prices ({col_list}) VALUES ({placeholders})"
            cursor.executemany(insert_sql, rows)
            inserted_count = cursor.rowcount
            
            # Update existing records for the same symbol and date
            update_cols = [col for col in expected_cols if col not in ['symbol', 'date']]  # Don't update key columns
            update_set = ", ".join([f"{col} = ?" for col in update_cols])
            
            update_sql = f"""
            UPDATE stock_prices 
            SET {update_set}
            WHERE symbol = ? AND date = ?
            """
            updated_count = 0
            for row in rows:
                row_dict = dict(zip(expected_cols, row))
                update_values = [row_dict[col] for col in update_cols]  # Values for SET clause
                update_values.extend([row_dict['symbol'], row_dict['date']])  # Values for WHERE clause
                
                cursor.execute(update_sql, update_values)
                if cursor.rowcount > 0:
                    updated_count += 1
            
            conn.commit()
            print(f"Inserted {inserted_count} new rows, updated {updated_count} existing rows in stock_prices")
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error saving price data to database: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if conn:
                conn.close()

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

    @staticmethod
    def create_options_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
        """Create options chain visualization"""
        if df.empty:
            return go.Figure().add_annotation(text="No options data available", 
                                            xref="paper", yref="paper",
                                            x=0.5, y=0.5, showarrow=False)
        
        # Create subplots for calls and puts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Call Volume vs Strike', 'Put Volume vs Strike',
                          'Call Open Interest vs Strike', 'Put Open Interest vs Strike'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Separate calls and puts
        calls_df = df[df['option_type'] == 'call'].copy()
        puts_df = df[df['option_type'] == 'put'].copy()
        
        # Colors for different expiration dates
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Plot call volume
        if not calls_df.empty:
            for i, exp_date in enumerate(calls_df['expiration_date'].unique()):
                exp_calls = calls_df[calls_df['expiration_date'] == exp_date]
                fig.add_trace(
                    go.Bar(
                        x=exp_calls['strike'],
                        y=exp_calls['volume'],
                        name=f'Calls {exp_date}',
                        marker_color=colors[i % len(colors)],
                        opacity=0.7
                    ),
                    row=1, col=1
                )
        
        # Plot put volume
        if not puts_df.empty:
            for i, exp_date in enumerate(puts_df['expiration_date'].unique()):
                exp_puts = puts_df[puts_df['expiration_date'] == exp_date]
                fig.add_trace(
                    go.Bar(
                        x=exp_puts['strike'],
                        y=exp_puts['volume'],
                        name=f'Puts {exp_date}',
                        marker_color=colors[i % len(colors)],
                        opacity=0.7
                    ),
                    row=1, col=2
                )
        
        # Plot call open interest
        if not calls_df.empty:
            for i, exp_date in enumerate(calls_df['expiration_date'].unique()):
                exp_calls = calls_df[calls_df['expiration_date'] == exp_date]
                fig.add_trace(
                    go.Scatter(
                        x=exp_calls['strike'],
                        y=exp_calls['open_interest'],
                        mode='lines+markers',
                        name=f'Call OI {exp_date}',
                        line=dict(color=colors[i % len(colors)]),
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # Plot put open interest
        if not puts_df.empty:
            for i, exp_date in enumerate(puts_df['expiration_date'].unique()):
                exp_puts = puts_df[puts_df['expiration_date'] == exp_date]
                fig.add_trace(
                    go.Scatter(
                        x=exp_puts['strike'],
                        y=exp_puts['open_interest'],
                        mode='lines+markers',
                        name=f'Put OI {exp_date}',
                        line=dict(color=colors[i % len(colors)]),
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title=f"{symbol.upper()} - Options Chain Analysis (Volume ≥ 1000)",
            height=800,
            template="plotly_white"
        )
        
        fig.update_xaxes(title_text="Strike Price", row=1, col=1)
        fig.update_xaxes(title_text="Strike Price", row=1, col=2)
        fig.update_xaxes(title_text="Strike Price", row=2, col=1)
        fig.update_xaxes(title_text="Strike Price", row=2, col=2)
        
        fig.update_yaxes(title_text="Volume", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=1, col=2)
        fig.update_yaxes(title_text="Open Interest", row=2, col=1)
        fig.update_yaxes(title_text="Open Interest", row=2, col=2)
        
        return fig

def human_readable(num):
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num/1_000:.2f}K"
    else:
        return str(num)


def main():
    # Initialize components
    db_manager = DatabaseManager()
    data_fetcher = StockDataFetcher(db_manager)
    visualizer = StockVisualizer()
    
    # Header
    st.markdown('<h1 class="main-header">📈 Advanced Stock Market Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    # st.sidebar.title("⚙️ Dashboard Settings")
    
    # Auto-refresh
    # auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (5 min)", value=False)
    # if auto_refresh:
    #     time.sleep(300)  # 5 minutes
    #     st.rerun()
    
    # # Manual refresh
    # if st.sidebar.button("🔄 Refresh Data Now"):
    #     st.cache_data.clear()
    #     st.rerun()
    
    # Data source selection
    # data_source = st.sidebar.radio(
    #     "📊 Data Source:",
    #     ["Live Data (Yahoo)", "Database Cache"]
    # )
    stocks_df = data_fetcher.fetch_most_active_stocks()
    
    # Fetch most active stocks
    # try:
    #     if data_source == "Live Data (Yahoo)":
    #         stocks_df = data_fetcher.fetch_most_active_stocks()
    #     else:
    #         with db_manager.get_connection() as conn:
    #             today = datetime.now().strftime('%Y-%m-%d')
    #             # today = "2025-08-27"
    #             stocks_df = pd.read_sql_query(
    #                 "SELECT * FROM most_active_stocks WHERE scrape_date = ? ORDER BY volume DESC",
    #                 conn,
    #                 params=(today,)
    #             )
        
    #     if stocks_df is None or stocks_df.empty:
    #         st.error("No stock data available. Please try refreshing or check your connection.")
    #         return
        
    # except Exception as e:
    #     st.error(f"Error loading stock data: {str(e)}")
    #     return
    
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
    
    # Remove unwanted columns - specify the columns you want to keep
    columns_to_remove = ['week_52_range', 'Region', 'Follow', 'scrape_date']
    # Filter to only include columns that actually exist in the DataFrame
    # available_columns = [col for col in columns_to_remove if col in display_df.columns]
    display_df = display_df.drop(columns=[col for col in columns_to_remove if col in display_df.columns])
    
    # volume column should be displayed in string human redable
    display_df['volume'] = display_df['volume'].apply(human_readable)
    
    # Add clickable functionality with on_select
    selected_rows = st.dataframe(
        display_df,
        use_container_width=True,
        height=400,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    # Check if a row was selected and get the symbol
    selected_symbol_from_table = None
    if selected_rows.selection.rows:
        selected_row_index = selected_rows.selection.rows[0]
        if selected_row_index < len(stocks_df):
            selected_symbol_from_table = stocks_df.iloc[selected_row_index]['symbol']
            st.info(f"📊 Selected: **{selected_symbol_from_table}** - Analysis will be shown below")
    
    # Stock Selection and Analysis
    st.subheader("📈 Individual Stock Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if 'symbol' in stocks_df.columns and not stocks_df['symbol'].empty:
            # Use selected symbol from table if available, otherwise use manual input
            default_symbol = selected_symbol_from_table if selected_symbol_from_table else stocks_df['symbol'].tolist()[0]
            selected_symbol = st.text_input(
                "Enter stock symbol for detailed analysis:",
                value=default_symbol,
                placeholder="Type stock symbol (e.g., AAPL, MSFT, TSLA)...",
                help="Enter any valid stock ticker symbol or click on a stock from the table above",
                key="stock_search_input"
            ).upper().strip()
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
    
    # Options Chain Analysis Section
    st.subheader("📊 Options Chain Analysis")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if 'symbol' in stocks_df.columns and not stocks_df['symbol'].empty:
            # Use selected symbol from table if available, otherwise use manual input
            default_options_symbol = selected_symbol_from_table if selected_symbol_from_table else stocks_df['symbol'].tolist()[0]
            options_symbol = st.text_input(
                "Enter stock symbol for options analysis:",
                value=default_options_symbol,
                placeholder="Type stock symbol (e.g., AAPL, MSFT, TSLA)...",
                help="Enter any valid stock ticker symbol for options analysis or click on a stock from the table above",
                key="options_stock_search_input"
            ).upper().strip()
        else:
            st.error("No stock symbols available for options analysis")
            options_symbol = None
    
    with col2:
        min_volume = st.number_input(
            "Minimum Volume:",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="Filter options by minimum volume"
        )
    
    with col3:
        data_source_options = st.radio(
            "Options Data Source:",
            ["Live Data", "Database Cache"],
            key="options_data_source"
        )
    
    # Options data fetching and display
    if options_symbol:
        col1, col2 = st.columns([3, 1])
        
        with col2:
            fetch_options = st.button("🔄 Fetch Options Data", key="fetch_options_btn")
        
        options_df = None
        
        if fetch_options or data_source_options == "Live Data":
            if data_source_options == "Live Data":
                options_df = data_fetcher.fetch_option_chain(options_symbol, min_volume)
            else:
                options_df = data_fetcher.get_options_from_db(options_symbol, min_volume)
        else:
            # Try to load from database first
            options_df = data_fetcher.get_options_from_db(options_symbol, min_volume)
            
            if options_df is None:
                st.info(f"No cached options data found for {options_symbol}. Click 'Fetch Options Data' to load live data.")
        
        if options_df is not None and not options_df.empty:
            # Display options metrics
            st.subheader(f"📈 Options Overview for {options_symbol.upper()}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_contracts = len(options_df)
                st.metric(
                    label="Total Contracts",
                    value=f"{total_contracts:,}"
                )
            
            with col2:
                total_volume = options_df['volume'].sum()
                st.metric(
                    label="Total Volume",
                    value=f"{total_volume:,}"
                )
            
            with col3:
                call_contracts = len(options_df[options_df['option_type'] == 'call'])
                put_contracts = len(options_df[options_df['option_type'] == 'put'])
                call_put_ratio = call_contracts / put_contracts if put_contracts > 0 else 0
                st.metric(
                    label="Call/Put Ratio",
                    value=f"{call_put_ratio:.2f}",
                    help=f"Calls: {call_contracts}, Puts: {put_contracts}"
                )
            
            with col4:
                avg_iv = options_df['implied_volatility'].mean()
                st.metric(
                    label="Avg Implied Volatility",
                    value=f"{avg_iv:.2%}" if pd.notna(avg_iv) else "N/A"
                )
            
            # Options visualization
            st.subheader("📊 Options Chain Visualization")
            options_fig = visualizer.create_options_chart(options_df, options_symbol)
            st.plotly_chart(options_fig, use_container_width=True)
            
            # Options data table
            st.subheader("📋 Options Data Table")
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                option_type_filter = st.selectbox(
                    "Filter by Option Type:",
                    ["All", "Calls", "Puts"],
                    key="option_type_filter"
                )
            
            with col2:
                if 'expiration_date' in options_df.columns:
                    exp_dates = options_df['expiration_date'].unique()
                    selected_exp = st.selectbox(
                        "Filter by Expiration:",
                        ["All"] + list(exp_dates),
                        key="exp_date_filter"
                    )
                else:
                    selected_exp = "All"
            
            # Apply filters
            filtered_df = options_df.copy()
            
            if option_type_filter != "All":
                filter_type = "call" if option_type_filter == "Calls" else "put"
                filtered_df = filtered_df[filtered_df['option_type'] == filter_type]
            
            if selected_exp != "All" and 'expiration_date' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['expiration_date'] == selected_exp]
            
            # Format display columns
            display_columns = [
                'contract_symbol', 'expiration_date', 'strike', 'option_type',
                'last_price', 'bid', 'ask', 'volume', 'open_interest', 
                'implied_volatility', 'in_the_money', 'current_stock_price', 'strike_vs_price_pct'
            ]
            
            # Select available columns
            available_columns = [col for col in display_columns if col in filtered_df.columns]
            display_df = filtered_df[available_columns].copy()
            
            # Format numeric columns
            if 'implied_volatility' in display_df.columns:
                display_df['implied_volatility'] = display_df['implied_volatility'].apply(
                    lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
                )
            
            # Format current stock price
            if 'current_stock_price' in display_df.columns:
                display_df['current_stock_price'] = display_df['current_stock_price'].apply(
                    lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
                )
            
            # Format strike vs price percentage
            if 'strike_vs_price_pct' in display_df.columns:
                display_df['strike_vs_price_pct'] = display_df['strike_vs_price_pct'].apply(
                    lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
                )
            
            # Format strike prices
            if 'strike' in display_df.columns:
                display_df['strike'] = display_df['strike'].apply(
                    lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
                )
            
            # Format last price, bid, ask
            for price_col in ['last_price', 'bid', 'ask']:
                if price_col in display_df.columns:
                    display_df[price_col] = display_df[price_col].apply(
                        lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
                    )
            
            # Sort by volume descending
            if 'volume' in display_df.columns:
                display_df = display_df.sort_values('volume', ascending=False)
            
            # Rename columns for better display
            column_rename_map = {
                'contract_symbol': 'Contract',
                'expiration_date': 'Expiry',
                'strike': 'Strike',
                'option_type': 'Type',
                'last_price': 'Last Price',
                'bid': 'Bid',
                'ask': 'Ask',
                'volume': 'Volume',
                'open_interest': 'Open Int',
                'implied_volatility': 'IV',
                'in_the_money': 'ITM',
                'current_stock_price': 'Stock Price',
                'strike_vs_price_pct': 'Strike vs Stock %'
            }
            
            # Apply column renaming
            display_columns_renamed = {}
            for col in display_df.columns:
                if col in column_rename_map:
                    display_columns_renamed[col] = column_rename_map[col]
                else:
                    display_columns_renamed[col] = col.replace('_', ' ').title()
            
            display_df = display_df.rename(columns=display_columns_renamed)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
            
            # Summary statistics
            with st.expander("📊 Options Summary Statistics"):
                if not filtered_df.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Volume Statistics:**")
                        st.write(f"• Total Volume: {filtered_df['volume'].sum():,}")
                        st.write(f"• Average Volume: {filtered_df['volume'].mean():.0f}")
                        st.write(f"• Max Volume: {filtered_df['volume'].max():,}")
                        
                        st.write("**Price Statistics:**")
                        if 'last_price' in filtered_df.columns:
                            st.write(f"• Average Last Price: ${filtered_df['last_price'].mean():.2f}")
                            st.write(f"• Price Range: ${filtered_df['last_price'].min():.2f} - ${filtered_df['last_price'].max():.2f}")
                    
                    with col2:
                        st.write("**Strike Price Analysis:**")
                        if 'strike' in filtered_df.columns:
                            st.write(f"• Strike Range: ${filtered_df['strike'].min():.2f} - ${filtered_df['strike'].max():.2f}")
                            st.write(f"• Most Active Strike: ${filtered_df.loc[filtered_df['volume'].idxmax(), 'strike']:.2f}")
                        
                        # New enhanced metrics
                        if 'current_stock_price' in filtered_df.columns and not filtered_df['current_stock_price'].isna().all():
                            current_price = filtered_df['current_stock_price'].iloc[0]
                            st.write(f"• Current Stock Price: ${current_price:.2f}")
                        
                        if 'strike_vs_price_pct' in filtered_df.columns:
                            st.write("**Strike vs Stock Price:**")
                            st.write(f"• Range: {filtered_df['strike_vs_price_pct'].min():+.1f}% to {filtered_df['strike_vs_price_pct'].max():+.1f}%")
                            avg_distance = filtered_df['strike_vs_price_pct'].abs().mean()
                            st.write(f"• Avg Distance from Stock Price: ±{avg_distance:.1f}%")
                        
                        st.write("**Open Interest:**")
                        if 'open_interest' in filtered_df.columns:
                            st.write(f"• Total Open Interest: {filtered_df['open_interest'].sum():,}")
                            st.write(f"• Average Open Interest: {filtered_df['open_interest'].mean():.0f}")
                else:
                    st.write("No data available for the selected filters.")
        
        elif options_df is not None and options_df.empty:
            st.warning(f"No options data with volume >= {min_volume} found for {options_symbol}")
        
        elif fetch_options:
            st.error(f"Could not fetch options data for {options_symbol}. The stock may not have options or there may be a connectivity issue.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "📊 **Data provided by Yahoo Finance** | "
        "🔄 **Real-time updates available** | "
        f"🕒 **Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**"
    )

if __name__ == "__main__":
    main()