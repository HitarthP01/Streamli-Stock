"""
Database Manager Module
Handles all database operations and connection management.
"""

import sqlite3
import logging
from typing import Optional

logger = logging.getLogger(__name__)

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
            
            conn.commit()
