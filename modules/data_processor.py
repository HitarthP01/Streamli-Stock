"""
Data Processing Module
Handles complex data processing operations including OHLCV fetching, 
historical data management, and technical indicator calculations.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        
    def save_active_stocks_to_db(self, df: pd.DataFrame):
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
            
    def _fetch_and_store_historical_data(self, cursor, table_name: str, symbol: str, days: int = 60):
        """Fetch and store historical data for a new stock."""
        try:
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
            
            # Calculate technical indicators
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
            ohlcv_data = self._fetch_ohlcv_data_with_retry(symbol, date)
            
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

    def _fetch_ohlcv_data_with_retry(self, symbol: str, date: str, retries: int = 3):
        """Fetch OHLCV data with retry logic"""
        for attempt in range(retries):
            try:
                return self._fetch_ohlcv_data(symbol, date)
            except Exception as e:
                if attempt < retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                    continue
                else:
                    logger.error(f"All attempts failed for {symbol}: {e}")
                    return None

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
