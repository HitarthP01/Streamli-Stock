"""
Stock Data Fetcher Module
Handles all data fetching operations from Yahoo Finance and other sources.
"""

import yfinance as yf
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import requests
import logging
from typing import Optional
import sys
import os

# Add parent directory to path to import store_data
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)



from .database_manager import DatabaseManager
from .data_processor import DataProcessor

logger = logging.getLogger(__name__)

class StockDataFetcher:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.data_processor = DataProcessor(db_manager)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9" 
        }
                   
    def fetch_most_active_stocks(self) -> Optional[pd.DataFrame]:
        """Fetch most active stocks with better error handling"""
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
                self.data_processor.save_active_stocks_to_db(df)
                
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

    def fetch_stock_history(self, symbol: str, period: str = "1mo") -> Optional[pd.DataFrame]:
        """Fetch historical stock data with technical indicators"""

        # first check symbol is in tracking table(stock_prices table) or not
        with self.db_manager.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT symbol FROM symbols_tracking  WHERE symbol = ?", (symbol.lower(),))
            row = cur.fetchone()
            df = pd.DataFrame()
            if row:
                logger.info(f"Symbol {symbol} found in tracking table.")
                # symbol exists, now fetch data only from the latest date onwards
                # get the latest date for the symbol
                cur.execute("SELECT MAX(date) FROM stock_prices WHERE symbol = ?", (symbol.lower(),))
                row = cur.fetchone()
                latest_date = None
                if row and row[0]:
                    latest_date = datetime.strptime(row[0], '%Y-%m-%d')
                    logger.info(f"Latest date for {symbol} in DB: {latest_date}")
                if latest_date:
                    # fetch data from latest_date + 1 day to today
                    start_date = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
                    stock = yf.Ticker(symbol)
                    df = stock.history(start=start_date, end=end_date)
            else:
                logger.info(f"Symbol {symbol} not found in tracking table. Adding it.")
                # insert symbol into tracking table
                cur.execute("INSERT INTO symbols_tracking (symbol, is_active) VALUES (?, ?)", (symbol.lower(), True))
                conn.commit()
                # symbol does not exist, fetch full period data
                stock = yf.Ticker(symbol)
                df = stock.history(period=period, interval='1d')
                
        # Reset index to make Date a column
        df.reset_index(inplace=True)
        
        # Standardize column names
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        df['symbol'] = symbol.lower()
        
        # Calculate technical indicators
        df = self._calculate_technical_indicators(df)
        
        # Save to database
        # if store_data:
        #     store_data.store_data(df)
        # else:
        #     # Fallback: save using basic method
        #     self._save_price_data_to_db(df)
        
        return df
                      
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
