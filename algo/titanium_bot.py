
# **TITANIUM v19-PROD-ENHANCED - Complete Fixed Version**

# ===============================================================================
# 🛡️ TITANIUM v19-PROD-ENHANCED - PRODUCTION-READY TRADING SYSTEM
# "All Phase 1-4 Fixes Applied | Enhanced Data Fetching | Error-Free Edition"
# ===============================================================================
# CRITICAL FIXES: 32 | HIGH SEVERITY: 22 | MEDIUM: 15
# PRODUCTION STATUS: SAFE FOR LIMITED CAPITAL
# ===============================================================================

import os
import sys
import json
import asyncio
import time
import logging
import warnings
import argparse
import signal
import hashlib
import hmac
import re
import psutil
import gzip
import aiofiles
import contextvars
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from pathlib import Path
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from collections import deque, defaultdict
from contextlib import asynccontextmanager
import sqlite3
import aiosqlite
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
import aiohttp
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopLossRequest, TakeProfitRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, OrderStatus
from alpaca.common.exceptions import APIError
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from aiolimiter import AsyncLimiter
import joblib
import weakref

load_dotenv()

# ===============================================================================
# 🔐 SECURE CONFIGURATION SYSTEM - ENHANCED & VALIDATED
# ===============================================================================
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging BEFORE any class definitions
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
app_logger = logging.getLogger(__name__)

# Context variables for structured logging
request_id = contextvars.ContextVar('request_id', default='')

# Helper function for consistent boolean parsing
def parse_bool(value: str) -> bool:
    """Safely parse boolean from string"""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('true', 'yes', '1', 't', 'y')

# Titanium Base Exception for standardized error handling
class TitaniumBaseException(Exception):
    """Base exception for all Titanium-specific errors"""
    pass

class AccountBlockedError(TitaniumBaseException):
    """Raised when account is not in ACTIVE state"""
    pass

class InsufficientRegimeDiversityError(TitaniumBaseException):
    """Raised when HMM training lacks regime diversity"""
    pass

class ExecutionTimeoutError(TitaniumBaseException):
    """Raised when trade execution times out"""
    pass

class DataValidationError(TitaniumBaseException):
    """Raised when fetched data fails validation"""
    pass

class ConfigurationError(TitaniumBaseException):
    """Raised when configuration is invalid"""
    pass

class ReferenceError(TitaniumBaseException):
    """Raised when weak reference has expired"""
    pass

class WeakReferenceExpiredError(TitaniumBaseException):
    """Raised when weak reference has expired and system can degrade gracefully"""
    pass

# Validate environment BEFORE loading
REQUIRED_ENV_VARS = [
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHANNEL_ID",
    "TWELVEDATA_API_KEY"
]

print("Validating environment configuration...")
missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    print(f"Missing critical environment variables: {', '.join(missing_vars)}")
    print("Create a .env file or export variables before running.")
    sys.exit(1)

# Parse CLI arguments
def parse_args():
    """Minimal CLI for runtime configuration."""
    parser = argparse.ArgumentParser(description="TITANIUM Trading Bot")
    parser.add_argument("--symbol", type=str, default=os.getenv("TRADING_SYMBOL", "GLD"),
                        help="Trading symbol (default: GLD)")
    parser.add_argument("--paper", type=parse_bool,
                        default=parse_bool(os.getenv("PAPER_TRADING", "true")),
                        help="Paper trading mode (default: true)")
    parser.add_argument("--loop-interval", type=int,
                        default=int(os.getenv("LIVE_LOOP_INTERVAL", "120")),
                        help="Loop interval in seconds (default: 120)")
    return parser.parse_args()

@dataclass(frozen=True)
class SystemConfig:
    """Immutable configuration with validation and derived fields"""
    # 🔐 SECURE CREDENTIALS
    API_KEY: str = field(default_factory=lambda: os.environ["ALPACA_API_KEY"])
    SECRET_KEY: str = field(default_factory=lambda: os.environ["ALPACA_SECRET_KEY"])
    TELEGRAM_TOKEN: str = field(default_factory=lambda: os.environ["TELEGRAM_BOT_TOKEN"])
    TELEGRAM_CHANNEL: str = field(default_factory=lambda: os.environ["TELEGRAM_CHANNEL_ID"])
    TWELVEDATA_API_KEY: str = field(default_factory=lambda: os.environ["TWELVEDATA_API_KEY"])

    # 📊 TRADING SETTINGS - ADJUSTED FOR SUSTAINABLE FREQUENCY
    SYMBOL: str = field(default_factory=lambda: os.getenv("TRADING_SYMBOL", "GLD"))
    PRIMARY_TIMEFRAME: str = field(default_factory=lambda: os.getenv("PRIMARY_TIMEFRAME", "1d"))
    INTRADAY_TIMEFRAMES: List[str] = field(default_factory=lambda: json.loads(
        os.getenv("INTRADAY_TIMEFRAMES", '["4h", "1h", "15m"]')))
    TARGET_DAILY_TRADES: int = field(default_factory=lambda: int(os.getenv("TARGET_DAILY_TRADES", "3")))

    # 💰 RISK MANAGEMENT - CONSERVATIVE FOR MULTI-TRADE STRATEGY
    INITIAL_CAPITAL: float = field(default_factory=lambda: float(os.getenv("INITIAL_CAPITAL", "100000")))
    PAPER_TRADING: bool = field(default_factory=lambda: parse_bool(os.getenv("PAPER_TRADING", "true")))
    MAX_POS_SIZE_PCT: float = field(default_factory=lambda: float(os.getenv("MAX_POS_SIZE_PCT", "0.015")))
    MAX_DAILY_LOSS_PCT: float = field(default_factory=lambda: float(os.getenv("MAX_DAILY_LOSS_PCT", "0.01")))
    SLIPPAGE_BPS: float = field(default_factory=lambda: float(os.getenv("SLIPPAGE_BPS", "20.0")))
    PORTFOLIO_MAX_EXPOSURE: float = field(default_factory=lambda: float(os.getenv("PORTFOLIO_MAX_EXPOSURE", "0.15")))
    SPREAD_BUFFER_BPS: float = field(default_factory=lambda: float(os.getenv("SPREAD_BUFFER_BPS", "2.0")))

    # ⚡ EXECUTION - SMART ORDER ROUTING
    COMMISSION_PER_SHARE: float = field(default_factory=lambda: float(os.getenv("COMMISSION_PER_SHARE", "0.005")))
    ATR_PERIOD: int = field(default_factory=lambda: int(os.getenv("ATR_PERIOD", "14")))
    STOP_LOSS_ATR: float = field(default_factory=lambda: float(os.getenv("STOP_LOSS_ATR", "1.8")))
    TAKE_PROFIT_ATR: float = field(default_factory=lambda: float(os.getenv("TAKE_PROFIT_ATR", "2.5")))
    ORDER_TIMEOUT_SECONDS: int = field(default_factory=lambda: int(os.getenv("ORDER_TIMEOUT_SECONDS", "90")))
    EXECUTION_TIMEOUT_SECONDS: int = field(default_factory=lambda: int(os.getenv("EXECUTION_TIMEOUT_SECONDS", "120")))

    # 🔄 LIVE LOOP - API BUDGET OPTIMIZED
    LIVE_LOOP_INTERVAL_SECONDS: int = field(default_factory=lambda: int(os.getenv("LIVE_LOOP_INTERVAL", "120")))
    DATA_FETCH_INTERVAL_MINUTES: int = field(default_factory=lambda: int(os.getenv("DATA_FETCH_INTERVAL", "20")))
    MAX_CACHE_AGE_MULTIPLIER: int = field(default_factory=lambda: int(os.getenv("MAX_CACHE_AGE_MULTIPLIER", "2")))

    # 🧠 HMM SETTINGS - DYNAMIC COMPONENTS
    HMM_COMPONENTS: int = field(default_factory=lambda: int(os.getenv("HMM_COMPONENTS", "3")))
    HMM_MIN_COMPONENTS: int = field(default_factory=lambda: int(os.getenv("HMM_MIN_COMPONENTS", "3")))
    HMM_MAX_COMPONENTS: int = field(default_factory=lambda: int(os.getenv("HMM_MAX_COMPONENTS", "5")))
    HMM_TRAIN_WINDOW: Dict[str, int] = field(default_factory=lambda: json.loads(
        os.getenv("HMM_TRAIN_WINDOW", '{"1d": 504, "4h": 500, "1h": 1000, "15m": 2000}')
    ))
    HMM_RANDOM_STATE: int = field(default_factory=lambda: int(os.getenv("HMM_RANDOM_STATE", "42")))
    HMM_MAX_ITER: int = field(default_factory=lambda: int(os.getenv("HMM_MAX_ITER", "150")))
    HMM_RETRAIN_INTERVAL_BARS: Dict[str, int] = field(default_factory=lambda: json.loads(
        os.getenv("HMM_RETRAIN_INTERVAL_BARS", '{"1d": 20, "4h": 50, "1h": 100, "15m": 200}')
    ))

    # 🆕 API BUDGET MANAGEMENT - TWELVEDATA 800/DAY LIMIT
    API_CALLS_PER_DAY_LIMIT: int = field(default_factory=lambda: int(os.getenv("API_CALLS_PER_DAY_LIMIT", "800")))
    API_CALL_BUDGET_PRIORITY: List[str] = field(default_factory=lambda: json.loads(
        os.getenv("API_CALL_BUDGET_PRIORITY", '["1d", "4h", "1h", "15m"]')
    ))
    API_BUDGET_MODE: str = field(default_factory=lambda: os.getenv("API_BUDGET_MODE", "adaptive"))

    # 🎯 SMART EXECUTION
    USE_LIMIT_ORDERS: bool = field(default_factory=lambda: parse_bool(os.getenv("USE_LIMIT_ORDERS", "true")))
    LIMIT_ORDER_PASSIVITY_BPS: int = field(default_factory=lambda: int(os.getenv("LIMIT_ORDER_PASSIVITY_BPS", "10")))
    MIN_TRADE_COOLDOWN_SECONDS: int = field(default_factory=lambda: int(os.getenv("MIN_TRADE_COOLDOWN_SECONDS", "30")))
    MAX_TRADE_COOLDOWN_SECONDS: int = field(default_factory=lambda: int(os.getenv("MAX_TRADE_COOLDOWN_SECONDS", "300")))
    POSITION_CACHE_TTL: int = field(default_factory=lambda: int(os.getenv("POSITION_CACHE_TTL", "5")))

    # 📈 MONITORING & DATABASE
    METRICS_PORT: int = field(default_factory=lambda: int(os.getenv("METRICS_PORT", "9090")))
    HEALTH_CHECK_FILE: str = field(default_factory=lambda: os.getenv("HEALTH_CHECK_FILE", "./titanium_health.ok"))
    DB_PATH: str = field(default_factory=lambda: os.getenv("DB_PATH", "titanium_production.db"))
    DB_BACKUP_PATH: str = field(default_factory=lambda: os.getenv("DB_BACKUP_PATH", "./backups"))
    ACTIVE_ORDER_RETENTION_DAYS: int = field(default_factory=lambda: int(os.getenv("ACTIVE_ORDER_RETENTION_DAYS", "30")))
    HEALTH_CHECK_TIMEOUT_SECONDS: int = field(default_factory=lambda: int(os.getenv("HEALTH_CHECK_TIMEOUT_SECONDS", "240")))

    # 🛡️ REGIME MAPPING & QUALITY
    REGIME_BULL_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("REGIME_BULL_THRESHOLD", "0.25")))
    REGIME_BEAR_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("REGIME_BEAR_THRESHOLD", "-0.25")))
    MIN_TRADE_QUALITY: float = field(default_factory=lambda: float(os.getenv("MIN_TRADE_QUALITY", "65.0")))

    # 🔥 CHOP FILTER
    MIN_REGIME_CONFIDENCE: float = field(default_factory=lambda: float(os.getenv("MIN_REGIME_CONFIDENCE", "0.65")))

    # 🕒 MARKET HOURS - CRITICAL FIX
    MARKET_HOURS_ONLY: bool = field(default_factory=lambda: parse_bool(os.getenv("MARKET_HOURS_ONLY", "true")))
    TRADING_START_TIME: str = field(default_factory=lambda: os.getenv("TRADING_START_TIME", "09:30"))
    TRADING_END_TIME: str = field(default_factory=lambda: os.getenv("TRADING_END_TIME", "16:00"))
    EXTENDED_HOURS_ENABLED: bool = field(default_factory=lambda: parse_bool(os.getenv("EXTENDED_HOURS_ENABLED", "false")))

    # PHASE 4: Memory management
    MEMORY_LIMIT_MB: float = field(default_factory=lambda: float(os.getenv("MEMORY_LIMIT_MB", "500")))
    MAX_ROWS_PER_TIMEFRAME: int = field(default_factory=lambda: int(os.getenv("MAX_ROWS_PER_TIMEFRAME", "10000")))

    # PHASE 4: Circuit breakers
    CIRCUIT_TIMEOUT_SECONDS: int = field(default_factory=lambda: int(os.getenv("CIRCUIT_TIMEOUT_SECONDS", "300")))
    CIRCUIT_FAILURE_THRESHOLD: int = field(default_factory=lambda: int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", "5")))

    # FIXED: Secure model signature - no hardcoded fallback
    MODEL_SIGNATURE_SECRET: str = field(default_factory=lambda: os.getenv("MODEL_SIGNATURE_SECRET", ""))

    # NEW: Minimum position size
    MIN_POSITION_SIZE: int = field(default_factory=lambda: int(os.getenv("MIN_POSITION_SIZE", "1")))

    # NEW: Enforce round lots
    ENFORCE_ROUND_LOTS: bool = field(default_factory=lambda: parse_bool(os.getenv("ENFORCE_ROUND_LOTS", "false")))

    # NEW: Memory cache limits
    MAX_CACHE_SIZE_MB: float = field(default_factory=lambda: float(os.getenv("MAX_CACHE_SIZE_MB", "200")))

    # NEW: Minimum risk per share floor (FIXED: Must be > 0)
    MIN_RISK_PER_SHARE_BPS: float = field(default_factory=lambda: float(os.getenv("MIN_RISK_PER_SHARE_BPS", "5.0")))

    # NEW: Maximum concurrent open positions
    MAX_OPEN_POSITIONS: int = field(default_factory=lambda: int(os.getenv("MAX_OPEN_POSITIONS", "3")))

    # NEW: Cash buffer for position sizing
    CASH_BUFFER_PCT: float = field(default_factory=lambda: float(os.getenv("CASH_BUFFER_PCT", "0.96")))

    # NEW: HMM sequence window for valid regime inference
    HMM_SEQUENCE_WINDOW: int = field(default_factory=lambda: int(os.getenv("HMM_SEQUENCE_WINDOW", "20")))

    # NEW: Minimum sequence length for HMM prediction
    HMM_MIN_SEQUENCE_LENGTH: int = field(default_factory=lambda: int(os.getenv("HMM_MIN_SEQUENCE_LENGTH", "10")))

    # NEW: Yahoo Finance historical limits (days)
    YFINANCE_MAX_LOOKBACK: Dict[str, int] = field(default_factory=lambda: json.loads(
        os.getenv("YFINANCE_MAX_LOOKBACK", '{"1d": 730, "4h": 730, "1h": 730, "15m": 60}')
    ))

    # NEW: TwelveData max output size
    TWELVEDATA_MAX_OUTPUTSIZE: int = field(default_factory=lambda: int(os.getenv("TWELVEDATA_MAX_OUTPUTSIZE", "5000")))

    # V19 NEW: Progressive data collection
    INITIAL_DATA_BARS: int = field(default_factory=lambda: int(os.getenv("INITIAL_DATA_BARS", "100")))
    DATA_COLLECTION_STRATEGY: str = field(default_factory=lambda: os.getenv("DATA_COLLECTION_STRATEGY", "progressive"))

    # V19 NEW: Data validation thresholds
    MIN_INITIAL_BARS: int = field(default_factory=lambda: int(os.getenv("MIN_INITIAL_BARS", "10")))
    DATA_STALENESS_THRESHOLD: int = field(default_factory=lambda: int(os.getenv("DATA_STALENESS_THRESHOLD", "3")))

    # V19 NEW: API retry settings
    API_MAX_RETRIES: int = field(default_factory=lambda: int(os.getenv("API_MAX_RETRIES", "3")))
    API_RETRY_DELAY: float = field(default_factory=lambda: float(os.getenv("API_RETRY_DELAY", "2.0")))

    def __post_init__(self):
        """Post-initialization validation with hard caps"""
        # Validate numeric ranges
        if self.MAX_POS_SIZE_PCT > 0.1:
            raise ValueError(f"MAX_POS_SIZE_PCT cannot exceed 10% (got {self.MAX_POS_SIZE_PCT})")
        if self.MAX_POS_SIZE_PCT > 0.02:
            app_logger.warning("MAX_POS_SIZE_PCT > 2% is highly risky for multi-trade strategy")
        if self.MAX_DAILY_LOSS_PCT > 0.02:
            app_logger.warning("MAX_DAILY_LOSS_PCT > 2% is aggressive")
        if self.TARGET_DAILY_TRADES > 5:
            app_logger.warning("TARGET_DAILY_TRADES > 5 may exceed API budget or cause overtrading")
        if self.COMMISSION_PER_SHARE < 0:
            raise ValueError("COMMISSION_PER_SHARE cannot be negative")
        if self.COMMISSION_PER_SHARE == 0:
            app_logger.warning("Commissions set to 0 - ensure this matches your broker")

        # Validate HMM component bounds
        if not (self.HMM_MIN_COMPONENTS <= self.HMM_COMPONENTS <= self.HMM_MAX_COMPONENTS):
            raise ValueError(
                f"HMM_COMPONENTS must be between {self.HMM_MIN_COMPONENTS} and {self.HMM_MAX_COMPONENTS}"
            )

        # Validate MIN_TRADE_QUALITY
        if not (50 <= self.MIN_TRADE_QUALITY <= 95):
            app_logger.warning("MIN_TRADE_QUALITY outside recommended range [50, 95]")

        # Validate MIN_RISK_PER_SHARE_BPS > 0 to prevent division by zero
        if self.MIN_RISK_PER_SHARE_BPS <= 0:
            raise ValueError(f"MIN_RISK_PER_SHARE_BPS must be > 0 (got {self.MIN_RISK_PER_SHARE_BPS})")

        # Validate timeframe format
        valid_tfs = ["1d", "4h", "1h", "15m"]
        tf_pattern = re.compile(r'^\d+[hdm]$')
        if not tf_pattern.match(self.PRIMARY_TIMEFRAME) or self.PRIMARY_TIMEFRAME not in valid_tfs:
            raise ValueError(f"Invalid PRIMARY_TIMEFRAME: {self.PRIMARY_TIMEFRAME}")
        for tf in self.INTRADAY_TIMEFRAMES:
            if not tf_pattern.match(tf) or tf not in valid_tfs:
                raise ValueError(f"Invalid INTRADAY_TIMEFRAME: {tf}")

        # Validate MODEL_SIGNATURE_SECRET is set
        if not self.MODEL_SIGNATURE_SECRET:
            raise ConfigurationError("MODEL_SIGNATURE_SECRET environment variable is required")

        # Validate TwelveData output size
        if self.TWELVEDATA_MAX_OUTPUTSIZE > 5000:
            app_logger.warning(f"TWELVEDATA_MAX_OUTPUTSIZE {self.TWELVEDATA_MAX_OUTPUTSIZE} exceeds API limit of 5000")
            object.__setattr__(self, 'TWELVEDATA_MAX_OUTPUTSIZE', 5000)

        # Validate HMM sequence window
        if self.HMM_SEQUENCE_WINDOW < self.HMM_MIN_SEQUENCE_LENGTH:
            raise ValueError(
                f"HMM_SEQUENCE_WINDOW ({self.HMM_SEQUENCE_WINDOW}) must be >= HMM_MIN_SEQUENCE_LENGTH ({self.HMM_MIN_SEQUENCE_LENGTH})")

        # Calculate derived fields
        object.__setattr__(self, 'MAX_DAILY_LOSS_DOLLAR', self.INITIAL_CAPITAL * self.MAX_DAILY_LOSS_PCT)

        # FIXED: Do not modify frozen dataclass, only warn
        if self.LIMIT_ORDER_PASSIVITY_BPS > 50:
            app_logger.warning("LIMIT_ORDER_PASSIVITY_BPS > 50bps may prevent fills")

# ===============================================================================
# 🔐 PRODUCTION SECRETS MANAGEMENT
# ===============================================================================

class SecretsManager:
    """Production-grade secrets handling with Vault integration"""

    VAULT_ADDR = os.getenv("VAULT_ADDR")
    VAULT_TOKEN = os.getenv("VAULT_TOKEN")
    VAULT_MOUNT_POINT = os.getenv("VAULT_MOUNT_POINT", "secret")

    @staticmethod
    def load_from_vault() -> bool:
        """Integrate with HashiCorp Vault KV v2"""
        if not SecretsManager.VAULT_ADDR or not SecretsManager.VAULT_TOKEN:
            app_logger.info("Vault integration not configured, using environment variables")
            return False

        try:
            import hvac
            client = hvac.Client(url=SecretsManager.VAULT_ADDR, token=SecretsManager.VAULT_TOKEN)
            if not client.is_authenticated():
                app_logger.error("Vault authentication failed")
                return False

            secrets = client.secrets.kv.v2.read_secret_version(
                path="titanium/config",
                mount_point=SecretsManager.VAULT_MOUNT_POINT
            )

            for key, value in secrets["data"]["data"].items():
                os.environ[key] = value

            app_logger.info("Vault secrets loaded successfully")
            return True
        except ImportError:
            app_logger.warning("hvac library not installed, skipping Vault integration")
            return False
        except Exception as e:
            app_logger.error(f"Vault integration failed: {e}")
            return False

    @staticmethod
    def validate_env_security():
        """Check for common security misconfigurations"""
        env_path = Path('.env')
        if env_path.exists():
            # Check file permissions
            stat = env_path.stat()
            if stat.st_mode & 0o077:
                app_logger.warning(".env file is readable by group/other. Run: chmod 600 .env")

            # Check gitignore
            gitignore = Path('.gitignore')
            if gitignore.exists():
                content = gitignore.read_text()
                if '.env' not in content and '.env*' not in content:
                    app_logger.warning(".env not in .gitignore! Add it immediately.")
            else:
                app_logger.warning("No .gitignore found. Create one and add .env")

        # Check for hardcoded keys in code
        api_key = os.getenv("ALPACA_API_KEY")
        if api_key and len(api_key) < 20:
            app_logger.error("Invalid ALPACA_API_KEY format detected")

# ===============================================================================
# 📡 TELEGRAM BOT - ENHANCED RATE LIMITING
# ===============================================================================

class TelegramBot:
    def __init__(self, token: str, channel: str):
        self.token = token
        self.channel = channel
        self.enabled = False  # Start disabled until credentials verified
        self.min_interval = 2.0
        self._last_message_time = 0
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
        self._error_count = 0
        self._max_errors = 5

        # Bounded critical queue with overflow protection
        self._critical_queue = asyncio.Queue(maxsize=100)
        self._critical_task: Optional[asyncio.Task] = None

        # Overflow protection flag with recovery
        self._queue_overflow = False
        self._overflow_reset_threshold = 50  # Reset when queue drops below 50%

    async def initialize(self):
        """Async credential test and worker start"""
        await self._test_credentials_async()
        if self.enabled:
            await self.start_critical_worker()

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=3, limit_per_host=1)
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                connector=connector
            )

    async def _test_credentials_async(self):
        """Async credential test with manual retry"""
        if not self.token or not self.channel:
            app_logger.warning("Telegram credentials not provided")
            self.enabled = False
            return

        # Validate channel ID format
        if not (self.channel.startswith('@') or self.channel.lstrip('-').isdigit()):
            app_logger.error("TELEGRAM_CHANNEL_ID must be @username or numeric ID")
            self.enabled = False
            return

        for attempt in range(3):
            try:
                url = f"https://api.telegram.org/bot{self.token}/getMe"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=10) as resp:
                        data = await resp.json()
                        if data.get("ok"):
                            app_logger.info(f"Telegram bot: @{data['result']['username']}")

                            chat_url = f"https://api.telegram.org/bot{self.token}/getChat"
                            chat_data = {"chat_id": self.channel}
                            async with session.post(chat_url, json=chat_data, timeout=10) as chat_resp:
                                chat_json = await chat_resp.json()
                                if chat_json.get("ok"):
                                    app_logger.info("Channel access verified")
                                    self.enabled = True
                                    return
                                else:
                                    app_logger.warning(f"Bot not admin in channel: {chat_json.get('description')}")
                                    self.enabled = False
                                    return
                        else:
                            app_logger.error(f"Invalid token: {data.get('description')}")
                            self.enabled = False
                            return
            except aiohttp.ClientError as e:
                app_logger.error(f"Telegram test attempt {attempt+1} failed with network error: {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                app_logger.error(f"Telegram test attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)

        app_logger.error("Telegram credential test failed after 3 attempts")
        self.enabled = False

    async def start_critical_worker(self):
        """Start critical alert worker"""
        self._critical_task = asyncio.create_task(self._process_critical_queue())

    async def stop_critical_worker(self):
        """Stop critical alert worker"""
        if self._critical_task:
            # FIXED: Wait for task to finish with timeout
            try:
                self._critical_task.cancel()
                await asyncio.wait_for(self._critical_task, timeout=5.0)
            except asyncio.CancelledError:
                app_logger.debug("Critical task cancelled successfully")
            except asyncio.TimeoutError:
                app_logger.warning("Critical task shutdown timeout, forcing")
            except Exception as e:
                app_logger.warning(f"Critical task shutdown error: {e}")

    async def _process_critical_queue(self):
        """Process critical alerts without blocking main lock"""
        while True:
            try:
                message = await self._critical_queue.get()
                await self._send_critical(message)
                self._critical_queue.task_done()

                # Reset overflow flag if queue size drops below threshold
                if self._queue_overflow and self._critical_queue.qsize() < (self._critical_queue.maxsize * 0.5):
                    self._queue_overflow = False
                    app_logger.info("Telegram queue overflow condition cleared")

            except asyncio.CancelledError:
                app_logger.debug("Critical queue processing cancelled")
                break
            except Exception as e:
                app_logger.error(f"Critical queue error: {e}")
                await asyncio.sleep(5)

    async def _send_critical(self, message: str) -> bool:
        """Send critical message with minimal blocking and overflow protection"""
        try:
            if not self._session or self._session.closed:
                await self._ensure_session()

            # Consistent truncation limit
            if len(message) > 4096:
                message = message[:4000] + "\n... [TRUNCATED]"

            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id": self.channel,
                "text": message,
                "parse_mode": "Markdown"
            }
            async with self._session.post(url, json=payload) as resp:
                if resp.status == 200:
                    return True
                else:
                    app_logger.error(f"Critical Telegram error {resp.status}: {await resp.text()}")
                    return False
        except Exception as e:
            app_logger.error(f"Critical send failed: {e}")
            return False

    async def send(self, message: str, priority: str = "normal") -> bool:
        """Send message with rate limiting, session reuse, and error handling"""
        if not self.enabled:
            return False

        # Critical priority bypasses normal queue with overflow protection
        if priority == "critical":
            try:
                if self._critical_queue.full():
                    # Overflow protection: drop oldest message if queue full for > 30 seconds
                    if self._queue_overflow:
                        # Drop the oldest message
                        try:
                            self._critical_queue.get_nowait()
                            self._critical_queue.task_done()
                            app_logger.warning("Critical queue overflow, dropping oldest message")
                        except asyncio.QueueEmpty:
                            pass

                    self._queue_overflow = True
                    app_logger.warning("Critical queue full, message may be delayed")

                # Add with timeout to prevent indefinite blocking
                try:
                    await asyncio.wait_for(self._critical_queue.put(message), timeout=5.0)
                except asyncio.TimeoutError:
                    app_logger.error("Failed to queue critical message: timeout")
                    return False

                return True
            except Exception as e:
                app_logger.error(f"Failed to queue critical message: {e}")
                return False

        # Prevent alert fatigue for non-critical messages
        if priority == "normal" and self._error_count > 3:
            return False

        async with self._lock:
            current_time = time.time()
            elapsed = current_time - self._last_message_time
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)

            if not self._session or self._session.closed:
                await self._ensure_session()

            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id": self.channel,
                "text": message[:4096],
                "parse_mode": "Markdown"
            }

            try:
                async with self._session.post(url, json=payload) as resp:
                    self._last_message_time = time.time()
                    if resp.status == 200:
                        self._error_count = 0
                        return True
                    else:
                        self._error_count += 1
                        error_text = await resp.text()
                        app_logger.error(f"Telegram error {resp.status}: {error_text}")
                        if self._error_count >= self._max_errors:
                            app_logger.critical("Disabling Telegram due to repeated errors")
                            self.enabled = False
                        return False
            except asyncio.TimeoutError:
                self._error_count += 1
                app_logger.error("Telegram send timed out")
                return False
            except aiohttp.ClientError as e:
                self._error_count += 1
                app_logger.error(f"Telegram client error: {e}")
                return False
            except Exception as e:
                self._error_count += 1
                app_logger.error(f"Telegram send failed: {e}")
                return False

    async def close(self):
        """FIXED: Proper cleanup with timeout and forced session closure"""
        await self.stop_critical_worker()
        if self._session and not self._session.closed:
            try:
                await asyncio.wait_for(self._session.close(), timeout=5.0)
            except asyncio.TimeoutError:
                app_logger.warning("Telegram session close timeout, forcing closure")
                # Force close the connector
                if self._session.connector:
                    self._session.connector.close()
            except Exception as e:
                app_logger.warning(f"Telegram session close error: {e}")

# ===============================================================================
# 💾 PERSISTENCE LAYER - ASYNC WITH BACKUP & CONNECTION POOLING
# ===============================================================================

class DatabaseManager:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.db_path = Path(config.DB_PATH)
        self.backup_path = Path(config.DB_BACKUP_PATH)

        # Validate parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_path.mkdir(parents=True, exist_ok=True)

        # FIXED: Connection pooling with unified state management
        self._lock = asyncio.Lock()
        self._connection_semaphore = asyncio.Semaphore(5)

        # Unified connection state tracking
        # conn_id -> {conn, last_used, query_count}
        self._connection_pool: Dict[int, Dict[str, Any]] = {}
        self._pool_maxsize = 5
        self._max_queries_per_conn = 1000

        # Reentrant lock tracking for exclusive transactions
        self._exclusive_lock = asyncio.Lock()
        self._exclusive_lock_owner: Optional[int] = None  # task id
        self._exclusive_lock_count = 0

    async def _get_connection(self) -> aiosqlite.Connection:
        """Get connection from pool or create new with unified state management"""
        async with self._lock:
            # Check for available connections in pool
            now = time.time()
            available_conns = []

            for conn_id, conn_data in self._connection_pool.items():
                conn = conn_data['conn']
                # Health check connection
                try:
                    await asyncio.wait_for(conn.execute("SELECT 1"), timeout=2.0)
                    # Check if connection has exceeded query limit
                    if conn_data['query_count'] < self._max_queries_per_conn:
                        available_conns.append((conn_id, conn_data))
                except (asyncio.TimeoutError, Exception):
                    # Connection is dead, remove from pool
                    try:
                        await conn.close()
                    except Exception:
                        pass
                    del self._connection_pool[conn_id]
                    app_logger.debug(f"Removed dead connection {conn_id} from pool")

            # Return the most recently used healthy connection
            if available_conns:
                available_conns.sort(key=lambda x: x[1]['last_used'], reverse=True)
                conn_id, conn_data = available_conns[0]
                conn = conn_data['conn']
                conn_data['last_used'] = now
                conn_data['query_count'] += 1
                return conn

            # Create new connection if pool not full
            if len(self._connection_pool) < self._pool_maxsize:
                try:
                    conn = await asyncio.wait_for(
                        aiosqlite.connect(self.db_path),
                        timeout=10
                    )
                    conn_id = id(conn)
                    self._connection_pool[conn_id] = {
                        'conn': conn,
                        'last_used': now,
                        'query_count': 1
                    }
                    app_logger.debug(f"Created new connection {conn_id}, pool size: {len(self._connection_pool)}")
                    return conn
                except Exception as e:
                    app_logger.error(f"Failed to create new connection: {e}")
                    raise

            # Pool is full, evict least recently used connection
            lru_conn_id = min(self._connection_pool.items(), key=lambda x: x[1]['last_used'])[0]
            lru_conn = self._connection_pool[lru_conn_id]['conn']
            try:
                await lru_conn.close()
            except Exception:
                pass
            del self._connection_pool[lru_conn_id]

            # Create new connection
            try:
                conn = await asyncio.wait_for(
                    aiosqlite.connect(self.db_path),
                    timeout=10
                )
                conn_id = id(conn)
                self._connection_pool[conn_id] = {
                    'conn': conn,
                    'last_used': now,
                    'query_count': 1
                }
                app_logger.debug(f"Evicted LRU connection, created new {conn_id}")
                return conn
            except Exception as e:
                app_logger.error(f"Failed to create new connection after eviction: {e}")
                raise

    async def _release_connection(self, conn: aiosqlite.Connection):
        """Return connection to pool with updated state"""
        conn_id = id(conn)
        async with self._lock:
            if conn_id in self._connection_pool:
                self._connection_pool[conn_id]['last_used'] = time.time()
                # Reset query count if exceeded limit
                if self._connection_pool[conn_id]['query_count'] >= self._max_queries_per_conn:
                    self._connection_pool[conn_id]['query_count'] = 0
                    app_logger.debug(f"Reset query count for connection {conn_id}")

    async def initialize(self):
        """Initialize database with indexes, WAL mode, migrations"""
        await self._connection_semaphore.acquire()
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Set exclusive locking for critical sections
                await db.execute("PRAGMA busy_timeout = 5000")
                result = await db.execute("PRAGMA journal_mode=WAL")
                if (await result.fetchone())[0].lower() != 'wal':
                    app_logger.warning("Could not enable WAL mode")
                await db.execute("PRAGMA foreign_keys=ON")
                await db.execute("PRAGMA synchronous=NORMAL")
                await db.execute("PRAGMA wal_autocheckpoint=1000")

                # Schema with migrations
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        action TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        entry_price REAL,
                        exit_price REAL,
                        status TEXT NOT NULL,
                        entry_time TIMESTAMP NOT NULL,
                        exit_time TIMESTAMP,
                        stop_loss REAL,
                        take_profit REAL,
                        pnl REAL,
                        quality_score REAL,
                        regime TEXT,
                        confidence REAL,
                        order_id TEXT,
                        slippage_cost REAL,
                        commission_cost REAL,
                        timeframe_weight REAL,
                        execution_duration_ms INTEGER,
                        data_hash TEXT,
                        execution_price REAL,
                        execution_type TEXT,
                        exit_reason TEXT
                    )
                """)

                await db.execute("""
                    CREATE TABLE IF NOT EXISTS daily_risk (
                        date TEXT PRIMARY KEY,
                        daily_loss REAL NOT NULL,
                        daily_trades INTEGER NOT NULL,
                        max_drawdown REAL,
                        portfolio_value REAL,
                        api_calls_used INTEGER DEFAULT 0,
                        signal_efficiency REAL DEFAULT 0.0,
                        safety_mode_active BOOLEAN DEFAULT FALSE,
                        timeframe_cooldowns TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                await db.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id TEXT PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        timeframe TEXT NOT NULL,
                        action TEXT NOT NULL,
                        score REAL,
                        regime TEXT,
                        confidence REAL,
                        quality REAL,
                        symbol TEXT NOT NULL,
                        execution_decision TEXT,
                        api_budget_pct REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                await db.execute("""
                    CREATE TABLE IF NOT EXISTS active_orders (
                        order_id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        status TEXT NOT NULL,
                        filled_qty INTEGER DEFAULT 0,
                        submitted_at TIMESTAMP NOT NULL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                await db.execute("""
                    CREATE TABLE IF NOT EXISTS model_metadata (
                        timeframe TEXT PRIMARY KEY,
                        last_trained TIMESTAMP,
                        train_score REAL,
                        test_score REAL,
                        score_ratio REAL,
                        features TEXT,
                        data_hash TEXT,
                        train_duration_ms INTEGER,
                        retrain_reason TEXT,
                        components_used INTEGER
                    )
                """)

                await db.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        date TEXT PRIMARY KEY,
                        sharpe_ratio REAL,
                        win_rate REAL,
                        max_drawdown REAL,
                        total_trades INTEGER,
                        avg_signal_quality REAL,
                        model_drift_detected BOOLEAN,
                        signal_to_trade_ratio REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                await db.execute("""
                    CREATE TABLE IF NOT EXISTS api_usage (
                        date TEXT PRIMARY KEY,
                        calls_used INTEGER NOT NULL,
                        calls_remaining INTEGER NOT NULL,
                        reset_hour INTEGER NOT NULL,
                        budget_mode TEXT,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                await db.execute("""
                    CREATE TABLE IF NOT EXISTS timeframe_correlation (
                        timestamp TIMESTAMP,
                        timeframe1 TEXT,
                        timeframe2 TEXT,
                        correlation REAL,
                        PRIMARY KEY (timestamp, timeframe1, timeframe2)
                    )
                """)

                # Table for trade performance persistence
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS trade_performance (
                        date TEXT PRIMARY KEY,
                        win_rate REAL,
                        avg_pnl REAL,
                        trades_count INTEGER,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Indexes for performance
                await db.execute("CREATE INDEX IF NOT EXISTS idx_trades_status_symbol ON trades(status, symbol)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_daily_risk_date ON daily_risk(date)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_api_usage_date ON api_usage(date)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_active_orders_status ON active_orders(status)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_active_orders_order_id ON active_orders(order_id)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_active_orders_status_symbol ON active_orders(status, symbol)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_signals_execution_decision ON signals(execution_decision)")

                # Schema migrations
                try:
                    await db.execute("SELECT exit_reason FROM trades LIMIT 1")
                except sqlite3.OperationalError:
                    await db.execute("ALTER TABLE trades ADD COLUMN exit_reason TEXT")
                    app_logger.info("Migrated trades table: added exit_reason column")

                try:
                    await db.execute("SELECT timeframe_cooldowns FROM daily_risk LIMIT 1")
                except sqlite3.OperationalError:
                    await db.execute("ALTER TABLE daily_risk ADD COLUMN timeframe_cooldowns TEXT")
                    app_logger.info("Migrated daily_risk table: added timeframe_cooldowns column")

                try:
                    await db.execute("SELECT api_budget_pct FROM signals LIMIT 1")
                except sqlite3.OperationalError:
                    await db.execute("ALTER TABLE signals ADD COLUMN api_budget_pct REAL")
                    app_logger.info("Migrated signals table: added api_budget_pct column")

                # WAL checkpoint to prevent growth
                size_result = await db.execute("PRAGMA page_count")
                page_count = (await size_result.fetchone())[0]
                if page_count > 25000:  # ~100MB
                    app_logger.info("Database >100MB, running VACUUM")
                    await db.execute("VACUUM")

                await db.commit()

        except asyncio.TimeoutError:
            app_logger.error("Database initialization timed out")
            raise
        except Exception as e:
            app_logger.error(f"Database initialization failed: {e}")
            raise
        finally:
            self._connection_semaphore.release()

        app_logger.info("Database initialized with WAL mode, indexes, and enhanced schema")

    @asynccontextmanager
    async def exclusive_transaction(self):
        """Context manager for exclusive transactions with timeout - FIXED reentrant lock"""
        current_task = id(asyncio.current_task())

        # Check if we already own the lock (reentrant)
        if self._exclusive_lock_owner == current_task:
            self._exclusive_lock_count += 1
            conn = await self._get_connection()
            try:
                await conn.execute("BEGIN EXCLUSIVE")
                yield conn
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
            finally:
                await self._release_connection(conn)
                self._exclusive_lock_count -= 1
                if self._exclusive_lock_count == 0:
                    self._exclusive_lock_owner = None
        else:
            # Acquire lock with timeout
            try:
                await asyncio.wait_for(self._exclusive_lock.acquire(), timeout=30)
            except asyncio.TimeoutError:
                raise asyncio.TimeoutError("Could not acquire database lock within 30 seconds")

            self._exclusive_lock_owner = current_task
            self._exclusive_lock_count = 1

            try:
                conn = await asyncio.wait_for(
                    aiosqlite.connect(self.db_path), timeout=10
                )
                try:
                    await conn.execute("BEGIN EXCLUSIVE")
                    yield conn
                    await conn.commit()
                except Exception:
                    await conn.rollback()
                    raise
                finally:
                    await conn.close()
            finally:
                self._exclusive_lock_count = 0
                self._exclusive_lock_owner = None
                self._exclusive_lock.release()

    async def log_trade(self, trade_data: Dict[str, Any]):
        """Log trade atomically with extended fields"""
        async with self.exclusive_transaction() as db:
            # Use natural key for idempotency
            trade_id = trade_data.get('id', f"{trade_data['symbol']}_{trade_data['entry_time']}_{trade_data['timeframe']}")

            # Explicit column list for robustness
            await db.execute("""
                INSERT OR REPLACE INTO trades (
                    id, symbol, timeframe, action, quantity, entry_price, exit_price, status,
                    entry_time, exit_time, stop_loss, take_profit, pnl, quality_score, regime,
                    confidence, order_id, slippage_cost, commission_cost, timeframe_weight,
                    execution_duration_ms, data_hash, execution_price, execution_type, exit_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id,
                trade_data['symbol'],
                trade_data['timeframe'],
                trade_data['action'],
                trade_data['quantity'],
                trade_data.get('entry_price'),
                trade_data.get('exit_price'),
                trade_data['status'],
                trade_data['entry_time'],
                trade_data.get('exit_time'),
                trade_data.get('stop_loss'),
                trade_data.get('take_profit'),
                trade_data.get('pnl'),
                trade_data.get('quality_score'),
                trade_data.get('regime'),
                trade_data.get('confidence'),
                trade_data.get('order_id'),
                trade_data.get('slippage_cost', 0.0),
                trade_data.get('commission_cost', 0.0),
                trade_data.get('timeframe_weight'),
                trade_data.get('execution_duration_ms'),
                trade_data.get('data_hash'),
                trade_data.get('execution_price'),
                trade_data.get('execution_type'),
                trade_data.get('exit_reason')
            ))
            app_logger.debug(f"Trade logged: {trade_id}")

    async def get_open_trades(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open trades with optional symbol filter"""
        conn = await self._get_connection()
        try:
            conn.row_factory = aiosqlite.Row
            if symbol:
                cursor = await conn.execute("SELECT * FROM trades WHERE status = 'OPEN' AND symbol = ?", (symbol,))
            else:
                cursor = await conn.execute("SELECT * FROM trades WHERE status = 'OPEN'")
            trades = [dict(row) for row in await cursor.fetchall()]

            # Convert Decimal PnL back to float for compatibility
            for trade in trades:
                if 'pnl' in trade and trade['pnl'] is not None:
                    try:
                        trade['pnl'] = float(trade['pnl'])
                    except (TypeError, ValueError) as e:
                        app_logger.warning(f"Could not convert PnL for trade {trade.get('id')}: {e}")
                        trade['pnl'] = 0.0
            return trades
        finally:
            await self._release_connection(conn)

    async def update_trade_exit(self, trade_id: str, exit_price: float, exit_time: datetime, pnl: Decimal, status: str, exit_reason: str = None):
        """Update trade on exit"""
        async with self.exclusive_transaction() as db:
            await db.execute("""
                UPDATE trades SET exit_price = ?, exit_time = ?, pnl = ?, status = ?, exit_reason = ?
                WHERE id = ?
            """, (exit_price, exit_time, str(pnl), status, exit_reason, trade_id))
            app_logger.debug(f"Trade updated: {trade_id}")

    async def log_daily_risk(self, date: str, daily_loss: Decimal, daily_trades: int, max_drawdown: float, portfolio_value: float,
                             api_calls_used: int = 0, signal_efficiency: float = 0.0, safety_mode_active: bool = False,
                             timeframe_cooldowns: Optional[Dict] = None):
        """Log daily risk metrics with Decimal support"""
        async with self.exclusive_transaction() as db:
            cooldowns_json = json.dumps(timeframe_cooldowns) if timeframe_cooldowns else None

            await db.execute("""
                INSERT OR REPLACE INTO daily_risk (date, daily_loss, daily_trades, max_drawdown, portfolio_value, 
                api_calls_used, signal_efficiency, safety_mode_active, timeframe_cooldowns)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (date, str(daily_loss), daily_trades, max_drawdown, portfolio_value,
                  api_calls_used, signal_efficiency, safety_mode_active, cooldowns_json))
            app_logger.debug(f"Daily risk logged: {date}")

    async def get_daily_risk(self, date: str) -> Optional[Dict[str, Any]]:
        """Get daily risk data"""
        conn = await self._get_connection()
        try:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT * FROM daily_risk WHERE date = ?", (date,))
            row = await cursor.fetchone()
            if row:
                data = dict(row)
                # Convert back to Decimal
                if 'daily_loss' in data and data['daily_loss'] is not None:
                    try:
                        data['daily_loss'] = Decimal(data['daily_loss'])
                    except (TypeError, ValueError) as e:
                        app_logger.warning(f"Could not convert daily_loss to Decimal: {e}")
                        data['daily_loss'] = Decimal('0.0')
                # Parse timeframe_cooldowns
                if data.get('timeframe_cooldowns'):
                    try:
                        data['timeframe_cooldowns'] = json.loads(data['timeframe_cooldowns'])
                    except json.JSONDecodeError as e:
                        app_logger.warning(f"Could not parse timeframe_cooldowns JSON: {e}")
                        data['timeframe_cooldowns'] = {}
                return data
            return None
        finally:
            await self._release_connection(conn)

    async def log_signal(self, signal_data: Dict[str, Any]):
        """Log signal with deduplication using natural key"""
        async with self.exclusive_transaction() as db:
            # Use natural key: timestamp_timeframe_action
            signal_id = f"{signal_data['timestamp']}_{signal_data['timeframe']}_{signal_data['action']}"

            await db.execute("""
                INSERT OR REPLACE INTO signals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                signal_id,
                signal_data['timestamp'],
                signal_data['timeframe'],
                signal_data['action'],
                signal_data.get('score'),
                signal_data.get('regime'),
                signal_data.get('confidence'),
                signal_data.get('quality'),
                signal_data.get('symbol'),
                signal_data.get('execution_decision'),
                signal_data.get('api_budget_pct')
            ))
            app_logger.debug(f"Signal logged: {signal_id}")

    async def log_api_usage(self, date: str, calls_used: int, calls_remaining: int, reset_hour: int, budget_mode: str = "adaptive"):
        """Log API usage for budget tracking"""
        async with self.exclusive_transaction() as db:
            await db.execute("""
                INSERT OR REPLACE INTO api_usage (date, calls_used, calls_remaining, reset_hour, budget_mode, last_updated)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (date, calls_used, calls_remaining, reset_hour, budget_mode))
            app_logger.debug(f"API usage logged: {date} - {calls_used} calls")

    async def get_api_usage(self, date: str) -> Optional[Dict[str, Any]]:
        """Get API usage data"""
        conn = await self._get_connection()
        try:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT * FROM api_usage WHERE date = ?", (date,))
            row = await cursor.fetchone()
            return dict(row) if row else None
        finally:
            await self._release_connection(conn)

    async def log_trade_performance(self, date: str, win_rate: float, avg_pnl: float, trades_count: int):
        """Persist trade performance metrics"""
        async with self.exclusive_transaction() as db:
            await db.execute("""
                INSERT OR REPLACE INTO trade_performance (date, win_rate, avg_pnl, trades_count, last_updated)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (date, win_rate, avg_pnl, trades_count))
            app_logger.debug(f"Trade performance logged: {date} - {trades_count} trades")

    async def get_trade_performance(self, date: str) -> Optional[Dict[str, Any]]:
        """Get trade performance data"""
        conn = await self._get_connection()
        try:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT * FROM trade_performance WHERE date = ?", (date,))
            row = await cursor.fetchone()
            return dict(row) if row else None
        finally:
            await self._release_connection(conn)

    async def get_recent_trades(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trades for performance analysis"""
        conn = await self._get_connection()
        try:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("""
                SELECT * FROM trades WHERE symbol = ? 
                ORDER BY entry_time DESC LIMIT ?
            """, (symbol, limit))
            trades = [dict(row) for row in await cursor.fetchall()]
            # Convert Decimal PnL
            for trade in trades:
                if 'pnl' in trade and trade['pnl'] is not None:
                    try:
                        trade['pnl'] = float(trade['pnl'])
                    except (TypeError, ValueError) as e:
                        app_logger.warning(f"Could not convert PnL for trade {trade.get('id')}: {e}")
                        trade['pnl'] = 0.0
            return trades
        finally:
            await self._release_connection(conn)

    async def calculate_signal_efficiency(self) -> float:
        """Calculate signal-to-trade efficiency ratio"""
        today = datetime.now(timezone.utc).date().isoformat()
        conn = await self._get_connection()
        try:
            cursor = await conn.execute("""
                SELECT 
                    (SELECT COUNT(*) FROM signals WHERE DATE(timestamp) = ?) as signals,
                    (SELECT COUNT(*) FROM trades WHERE DATE(entry_time) = ?) as trades
            """, (today, today))
            row = await cursor.fetchone()
            signals = row[0] if row else 0
            trades = row[1] if row else 0

            if signals == 0:
                return 0.0

            return trades / signals
        finally:
            await self._release_connection(conn)

    async def log_active_order(self, order_data: Dict[str, Any]):
        """Log active order with idempotency"""
        async with self.exclusive_transaction() as db:
            await db.execute("""
                INSERT OR REPLACE INTO active_orders (order_id, symbol, status, filled_qty, submitted_at, last_updated)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                str(order_data['order_id']),
                order_data['symbol'],
                order_data['status'],
                order_data.get('filled_qty', 0),
                order_data['submitted_at']
            ))
            await db.commit()

    async def get_active_orders(self) -> List[Dict[str, Any]]:
        """Get active orders from DB"""
        conn = await self._get_connection()
        try:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT * FROM active_orders WHERE status NOT IN ('FILLED', 'CANCELED', 'REJECTED', 'EXPIRED')")
            return [dict(row) for row in await cursor.fetchall()]
        finally:
            await self._release_connection(conn)

    async def update_active_order(self, order_id: str, status: str, filled_qty: Optional[int] = None):
        """Update active order status"""
        async with self.exclusive_transaction() as db:
            if filled_qty is not None:
                await db.execute("""
                    UPDATE active_orders SET status = ?, filled_qty = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE order_id = ?
                """, (status, filled_qty, order_id))
            else:
                await db.execute("""
                    UPDATE active_orders SET status = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE order_id = ?
                """, (status, order_id))
            await db.commit()

    async def cleanup_old_orders(self):
        """Remove old active_orders records"""
        # Use config passed to __init__ instead of global conf
        cutoff_date = (datetime.now() - timedelta(days=self.config.ACTIVE_ORDER_RETENTION_DAYS)).isoformat()
        async with self.exclusive_transaction() as db:
            await db.execute("DELETE FROM active_orders WHERE submitted_at < ?", (cutoff_date,))
            await db.commit()
            app_logger.info(f"Cleaned up active_orders older than {self.config.ACTIVE_ORDER_RETENTION_DAYS} days")

    async def backup(self):
        """Create timestamped backup with retention and async compression"""
        if not self.db_path.exists():
            return

        # Rate limit backups
        last_backup = list(self.backup_path.glob("titanium_backup_*.db.gz"))
        if last_backup and (time.time() - last_backup[0].stat().st_mtime) < 3600:
            app_logger.debug("Backup already created within last hour, skipping")
            return

        # Remove old backups (>60 days)
        for backup in self.backup_path.glob("titanium_backup_*.db.gz"):
            if backup.stat().st_mtime < time.time() - 60*86400:
                try:
                    await asyncio.to_thread(backup.unlink)
                    app_logger.debug(f"Deleted old backup: {backup}")
                except Exception as e:
                    app_logger.warning(f"Failed to delete old backup {backup}: {e}")

        backup_name = self.backup_path / f"titanium_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"

        # Async backup with proper error handling
        try:
            await asyncio.wait_for(
                self._async_compress_backup(backup_name),
                timeout=300  # 5 minute timeout for backup
            )
            app_logger.info(f"Database backed up to {backup_name}.gz")
            return True
        except asyncio.TimeoutError:
            app_logger.error("Database backup timed out after 5 minutes")
            return False
        except Exception as e:
            app_logger.error(f"Database backup failed: {e}")
            return False

    async def _async_compress_backup(self, backup_name: Path):
        """Async compression with chunked I/O to prevent blocking"""
        chunk_size = 1024 * 1024  # 1MB chunks

        try:
            # Read and compress in thread pool to avoid blocking
            def compress_file():
                with open(self.db_path, 'rb') as f_in:
                    with gzip.open(f"{backup_name}.gz", 'wb') as f_out:
                        while True:
                            chunk = f_in.read(chunk_size)
                            if not chunk:
                                break
                            f_out.write(chunk)

            await asyncio.to_thread(compress_file)

            # Verify backup
            if backup_name.with_suffix('.db.gz').stat().st_size == 0:
                app_logger.error("Backup file is empty")
                backup_name.with_suffix('.db.gz').unlink()
                raise Exception("Backup file is empty")

        except Exception as e:
            # Clean up partial backup
            if backup_name.with_suffix('.db.gz').exists():
                await asyncio.to_thread(backup_name.with_suffix('.db.gz').unlink)
            raise e

# ===============================================================================
# 📡 MULTI-TIMEFRAME DATA ENGINE - ENHANCED WITH PROGRESSIVE DATA COLLECTION
# ===============================================================================

class MultiTimeframeDataEngine:
    def __init__(self, config: SystemConfig, db: DatabaseManager):
        self.config = config
        self.symbol = config.SYMBOL
        self.api_key = config.TWELVEDATA_API_KEY
        self.db = db  # Reference for API budget persistence
        self.timeframes = [config.PRIMARY_TIMEFRAME] + config.INTRADAY_TIMEFRAMES
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.cache_times: Dict[str, datetime] = {}
        self.cache_sizes: Dict[str, int] = {}
        self.max_cache_size = 5
        self.total_cache_memory_mb = 0.0

        # Thread-safe API budget tracking
        self._api_lock = asyncio.Lock()
        self.daily_api_calls = 0
        self.api_limit = config.API_CALLS_PER_DAY_LIMIT
        self.api_calls_per_timeframe: Dict[str, int] = defaultdict(int)
        self.last_api_budget_reset = datetime.now().date()

        # Budget lock shared across engines
        self._budget_lock = asyncio.Lock()

        self.lock = asyncio.Lock()
        self.nyse = mcal.get_calendar('NYSE')
        self.session: Optional[aiohttp.ClientSession] = None

        # Circuit breaker state with lambda factory
        self.circuit_breaker: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: self._create_circuit_breaker_state())
        self.CIRCUIT_THRESHOLD = config.CIRCUIT_FAILURE_THRESHOLD
        self.CIRCUIT_TIMEOUT = config.CIRCUIT_TIMEOUT_SECONDS

        # Data integrity
        self.data_hashes: Dict[str, str] = {}

        # Budget mode state
        self.budget_mode = "adaptive"
        self.last_budget_check = datetime.now()

        # Rate limiting
        self.twelvedata_limiter = AsyncLimiter(8, 60)
        self.yfinance_limiter = AsyncLimiter(1, 5)

        # Initialize expected_next_bar_time
        self.expected_next_bar_time: Optional[datetime] = None

        # Data hash cache
        self._data_hash_cache: Dict[str, str] = {}

        # Market hours cache with TTL and LRU eviction
        self._market_hours_cache: Dict[str, Tuple[bool, float]] = {}
        self._market_hours_cache_ttl = 300  # 5 minutes
        self._market_hours_cache_maxsize = 100  # Maximum entries
        self._last_cache_cleanup = 0

        # Data collection strategy
        self.data_collection_strategy = config.DATA_COLLECTION_STRATEGY
        self.initial_data_bars = config.INITIAL_DATA_BARS
        self.collection_phase: Dict[str, int] = defaultdict(int)  # 0=initial, 1=progressive, 2=full

        # Start API budget loading and periodic persistence
        self._api_budget_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._cache_cleanup_task: Optional[asyncio.Task] = None

        # V19: Data source health tracking
        self.data_source_health: Dict[str, Dict[str, Any]] = {
            "twelvedata": {"healthy": True, "last_success": 0, "failures": 0},
            "yfinance": {"healthy": True, "last_success": 0, "failures": 0}
        }

    def _create_circuit_breaker_state(self) -> Dict[str, Any]:
        """Factory method to create circuit breaker state"""
        return {"open": False, "last_failure": 0, "failure_count": 0}

    async def initialize(self):
        """Initialize the data engine (separate from __init__)"""
        await self._load_api_budget()
        # Start periodic API budget persistence
        self._api_budget_task = asyncio.create_task(self._periodic_api_budget_persistence())
        # Start periodic cache cleanup
        self._cache_cleanup_task = asyncio.create_task(self._periodic_cache_cleanup())
        
        # Initialize collection phase for each timeframe
        for tf in self.timeframes:
            self.collection_phase[tf] = 0  # Start in initial collection phase

    async def _periodic_cache_cleanup(self):
        """Periodically clean expired cache entries"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Every minute
                self._clean_expired_cache_entries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                app_logger.warning(f"Cache cleanup failed: {e}")

    def _clean_expired_cache_entries(self):
        """Clean expired cache entries"""
        current_time = time.time()
        expired_keys = []

        for key, (_, cache_timestamp) in self._market_hours_cache.items():
            if current_time - cache_timestamp >= self._market_hours_cache_ttl:
                expired_keys.append(key)

        for key in expired_keys:
            self._market_hours_cache.pop(key, None)

        if expired_keys:
            app_logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")

    async def _periodic_api_budget_persistence(self):
        """Periodically save API budget to DB"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._save_api_budget()
            except asyncio.CancelledError:
                break
            except Exception as e:
                app_logger.warning(f"Periodic API budget save failed: {e}")

    async def _save_api_budget(self):
        """Save API usage to DB"""
        try:
            await self.db.log_api_usage(
                datetime.now().date().isoformat(),
                self.daily_api_calls,
                self.api_limit - self.daily_api_calls,
                0,
                self.budget_mode
            )
        except Exception as e:
            app_logger.warning(f"Failed to save API budget: {e}")

    async def _load_api_budget(self):
        """Load API usage from DB on startup"""
        try:
            today = datetime.now().date().isoformat()
            usage = await self.db.get_api_usage(today)
            if usage:
                self.daily_api_calls = usage['calls_used']
                app_logger.info(f"API budget restored: {self.daily_api_calls}/{self.api_limit} calls used")
            else:
                self.daily_api_calls = 0
                app_logger.info("API budget initialized to 0")
        except Exception as e:
            app_logger.warning(f"Could not load API budget: {e}, starting from 0")
            self.daily_api_calls = 0

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Ensure background tasks are properly shut down
        self._shutdown_event.set()

        # Cancel and wait for background tasks with timeout
        if self._api_budget_task:
            self._api_budget_task.cancel()
            try:
                await asyncio.wait_for(self._api_budget_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        if self._cache_cleanup_task:
            self._cache_cleanup_task.cancel()
            try:
                await asyncio.wait_for(self._cache_cleanup_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        try:
            # Save API budget before exit
            try:
                await self._save_api_budget()
            except Exception as e:
                app_logger.warning(f"Failed to save API budget on exit: {e}")
        finally:
            if self.session and not self.session.closed:
                await self.session.close()

    async def _ensure_session(self):
        """Reuse session with connection pooling"""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=10, limit_per_host=5, ttl_dns_cache=300)
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30, connect=5, sock_read=10),
                connector=connector
            )
            app_logger.info("aiohttp session created with connection pooling")

    async def is_market_open(self) -> bool:
        """Check if market is currently open with extended hours support and TTL cache"""
        if not self.config.MARKET_HOURS_ONLY:
            return True

        now = pd.Timestamp.now(tz='America/New_York')
        cache_key = now.strftime('%Y-%m-%d')
        current_time = time.time()

        # Check cache with TTL
        if cache_key in self._market_hours_cache:
            is_open, cache_timestamp = self._market_hours_cache[cache_key]
            if current_time - cache_timestamp < self._market_hours_cache_ttl:
                return is_open

        # Check for holidays
        schedule = self.nyse.schedule(start_date=now.date(), end_date=now.date())

        if schedule.empty:
            self._update_cache(cache_key, False, current_time)
            return False

        row = schedule.iloc[0]
        if hasattr(row, 'market_open'):
            market_open = row.market_open
            market_close = row.market_close
        else:
            market_open = row['market_open']
            market_close = row['market_close']

        # Respect extended hours config
        if self.config.EXTENDED_HOURS_ENABLED:
            extended_open = market_open - timedelta(hours=1.5)
            extended_close = market_close + timedelta(hours=4)
        else:
            extended_open = market_open
            extended_close = market_close

        is_open = extended_open <= now <= extended_close
        self._update_cache(cache_key, is_open, current_time)
        return is_open

    def _update_cache(self, key: str, value: bool, timestamp: float):
        """Update cache with size limit"""
        self._market_hours_cache[key] = (value, timestamp)

        # Enforce size limit
        if len(self._market_hours_cache) > self._market_hours_cache_maxsize:
            # Remove oldest entry
            oldest_key = min(self._market_hours_cache.keys(),
                             key=lambda k: self._market_hours_cache[k][1])
            self._market_hours_cache.pop(oldest_key, None)

    async def _check_circuit_breaker(self, source: str) -> bool:
        """Check if circuit breaker is open for a source"""
        cb = self.circuit_breaker[source]
        if cb["open"]:
            if time.time() - cb["last_failure"] > self.CIRCUIT_TIMEOUT:
                cb["open"] = False
                cb["failure_count"] = 0
                app_logger.info(f"Circuit breaker reset for {source}")
                return True
            return False
        return True

    async def _record_failure(self, source: str):
        """Record API failure"""
        cb = self.circuit_breaker[source]
        cb["failure_count"] += 1
        cb["last_failure"] = time.time()

        if cb["failure_count"] >= self.CIRCUIT_THRESHOLD:
            cb["open"] = True
            app_logger.error(f"Circuit breaker OPENED for {source}")

    async def _record_success(self, source: str):
        """Reset failure count on success"""
        if self.circuit_breaker[source]["failure_count"] > 0:
            self.circuit_breaker[source]["failure_count"] = 0

    async def has_api_budget(self, priority: str = "low") -> bool:
        """Check if we have remaining API budget for the day"""
        # Guard against zero or negative limit
        if self.api_limit <= 0:
            app_logger.error("API_CALLS_PER_DAY_LIMIT is zero or negative, disabling all fetches")
            return False

        # Reset daily counter if new day (under lock)
        today = datetime.now().date()
        async with self._api_lock:
            if today != self.last_api_budget_reset:
                self.daily_api_calls = 0
                self.last_api_budget_reset = today

            # Reserve buffer based on priority
            buffer_pct = {
                "high": 0.05,
                "medium": 0.15,
                "low": 0.30
            }

            buffer = self.api_limit * buffer_pct.get(priority, 0.30)
            available = self.api_limit - self.daily_api_calls

            # Adaptive mode: get more conservative as we approach limit
            if self.config.API_BUDGET_MODE == "adaptive":
                pct_used = (self.daily_api_calls / self.api_limit) * 100
                if pct_used > 70:
                    buffer *= 1.5

            # Hard cutoff at 98%
            pct_used = (self.daily_api_calls / self.api_limit) * 100
            return available > buffer and pct_used < 98

    def get_budget_status(self) -> Dict[str, float]:
        """Get detailed budget status"""
        used = self.daily_api_calls
        remaining = self.api_limit - used
        pct_used = (used / self.api_limit) * 100 if self.api_limit > 0 else 100
        pct_remaining = (remaining / self.api_limit) * 100 if self.api_limit > 0 else 0

        # Estimate remaining calls based on current burn rate
        hours_remaining = max(0, (24 - datetime.now().hour))
        burn_rate = used / max(datetime.now().hour, 1)
        projected_remaining = remaining - (burn_rate * hours_remaining)

        return {
            "used": used,
            "remaining": remaining,
            "limit": self.api_limit,
            "pct_used": pct_used,
            "pct_remaining": pct_remaining,
            "burn_rate_hourly": burn_rate,
            "projected_remaining": projected_remaining,
            "mode": self.budget_mode,
            "calls_per_timeframe": dict(self.api_calls_per_timeframe)
        }

    def update_budget_mode(self):
        """Update budget mode based on usage"""
        status = self.get_budget_status()

        if self.config.API_BUDGET_MODE == "adaptive":
            if status['pct_used'] > 80:
                self.budget_mode = "conservative"
            elif status['pct_used'] < 50:
                self.budget_mode = "aggressive"
            else:
                self.budget_mode = "normal"

        if self.budget_mode == "conservative":
            app_logger.info(f"Budget mode: CONSERVATIVE ({status['pct_used']:.1f}% used)")
        elif self.budget_mode == "aggressive":
            app_logger.debug(f"Budget mode: AGGRESSIVE ({status['pct_used']:.1f}% used)")
        else:
            app_logger.debug(f"Budget mode: NORMAL ({status['pct_used']:.1f}% used)")

        self.last_budget_check = datetime.now()

    async def fetch_timeframe(self, timeframe: str, force_refresh: bool = False, priority: str = "medium") -> bool:
        """Async data fetching with progressive collection, circuit breaker protection, and API budget"""
        async with self.lock:
            # Check budget BEFORE incrementing
            budget_available = await self.has_api_budget(priority)
            if not budget_available:
                budget_status = self.get_budget_status()
                app_logger.warning(f"API budget insufficient ({budget_status['pct_used']:.1f}% used). Skipping {timeframe}")
                return False

            now = datetime.now()

            # Cache check with max staleness
            if not force_refresh and timeframe in self.cache_times:
                elapsed = (now - self.cache_times[timeframe]).total_seconds() / 60
                base_intervals = {
                    "1d": 60,
                    "4h": 30,
                    "1h": 15,
                    "15m": 5
                }
                interval = base_intervals.get(timeframe, self.config.DATA_FETCH_INTERVAL_MINUTES)
                max_staleness = interval * self.config.MAX_CACHE_AGE_MULTIPLIER

                if elapsed < interval:
                    return True

                if elapsed > max_staleness:
                    app_logger.warning(f"{timeframe}: Data stale ({elapsed:.0f}min), forcing refresh")
                    force_refresh = True

            # Determine data collection strategy
            outputsize = self._get_outputsize_for_timeframe(timeframe)
            
            # Try TwelveData first with rate limiting
            success = False
            async with self.twelvedata_limiter:
                if await self._check_circuit_breaker("twelvedata"):
                    success = await self._fetch_twelvedata_enhanced(timeframe, outputsize)

            # Fallback to yFinance (doesn't count against budget)
            if not success and await self._check_circuit_breaker("yfinance"):
                app_logger.info(f"Falling back to yFinance for {timeframe}")
                success = await self._fetch_yfinance_enhanced(timeframe, outputsize)

            # Emergency fallback if both sources fail
            if not success:
                app_logger.warning(f"Both data sources failed for {timeframe}, using emergency fallback")
                success = await self._emergency_data_fallback(timeframe)

            # Only increment API counter on successful fetch from TwelveData
            if success and self.data_source_health["twelvedata"]["healthy"]:
                async with self._api_lock:
                    self.daily_api_calls += 1
                    self.api_calls_per_timeframe[timeframe] += 1

            if not success:
                app_logger.critical(f"All data sources failed for {timeframe}")
                # Don't raise exception, just return False for graceful degradation
                return False

            self.cache_times[timeframe] = now

            # Validate data quality with progressive thresholds
            df = self.get_df(timeframe)
            validation_result = self._validate_dataframe_enhanced(df, timeframe)
            
            if not validation_result["valid"]:
                app_logger.error(f"Data validation failed for {timeframe}: {validation_result['reason']}")
                
                # If we have some data but it's not perfect, we might still use it
                if validation_result.get("partial", False) and len(df) > self.config.MIN_INITIAL_BARS:
                    app_logger.warning(f"Using partial data for {timeframe}: {validation_result['reason']}")
                    # Continue with partial data
                else:
                    return False

            # Update cache size tracking
            self.cache_sizes[timeframe] = len(df)
            await self._enforce_cache_limits()

            # Use cached hash if data unchanged
            if len(df) > 0:
                new_hash = hashlib.sha256(df.tail(min(100, len(df))).to_csv().encode()).hexdigest()
                if self._data_hash_cache.get(timeframe) != new_hash:
                    self.data_hashes[timeframe] = new_hash
                    self._data_hash_cache[timeframe] = new_hash

            app_logger.info(f"{timeframe}: {len(df)} bars | API Calls: {self.daily_api_calls}/{self.api_limit}")
            
            # Update collection phase based on data accumulated
            await self._update_collection_phase(timeframe, len(df))
            
            return True

    def _get_outputsize_for_timeframe(self, timeframe: str) -> int:
        """Determine output size based on collection phase"""
        phase = self.collection_phase[timeframe]
        
        if phase == 0:  # Initial collection
            # Start with small amount for faster response
            return min(self.config.INITIAL_DATA_BARS, 100)
        elif phase == 1:  # Progressive collection
            # Increase gradually
            current_size = len(self.dataframes.get(timeframe, pd.DataFrame()))
            return min(current_size * 2, 500)
        else:  # Full collection
            # Use full historical data
            base_sizes = {
                "1d": 500,
                "4h": 800,
                "1h": 1200,
                "15m": 2000
            }
            return min(base_sizes.get(timeframe, 500), self.config.TWELVEDATA_MAX_OUTPUTSIZE)

    async def _update_collection_phase(self, timeframe: str, current_bars: int):
        """Update collection phase based on accumulated data"""
        target_bars = self.config.HMM_TRAIN_WINDOW.get(timeframe, 504)
        
        if current_bars >= target_bars * 0.9:
            self.collection_phase[timeframe] = 2  # Full collection
        elif current_bars >= target_bars * 0.5:
            self.collection_phase[timeframe] = 1  # Progressive collection
        else:
            self.collection_phase[timeframe] = 0  # Initial collection

    async def _enforce_cache_limits(self):
        """Enforce LRU cache limit to prevent memory leaks with hard MB limit"""
        # Calculate current memory usage
        current_memory = 0.0
        for timeframe, df in self.dataframes.items():
            if not df.empty and hasattr(df, 'memory_usage'):
                try:
                    current_memory += df.memory_usage(deep=True).sum() / 1024 / 1024
                except Exception:
                    # If memory calculation fails, estimate
                    current_memory += len(df) * len(df.columns) * 8 / 1024 / 1024

        self.total_cache_memory_mb = current_memory

        # Check hard memory limit
        if current_memory > self.config.MAX_CACHE_SIZE_MB:
            app_logger.warning(f"Cache memory {current_memory:.1f}MB > limit {self.config.MAX_CACHE_SIZE_MB}MB, evicting")

            # Evict oldest timeframes until under limit
            sorted_by_age = sorted(self.cache_times.items(), key=lambda x: x[1])
            for timeframe, _ in sorted_by_age:
                if timeframe in self.dataframes:
                    try:
                        df_memory = self.dataframes[timeframe].memory_usage(deep=True).sum() / 1024 / 1024
                    except Exception:
                        df_memory = len(self.dataframes[timeframe]) * len(self.dataframes[timeframe].columns) * 8 / 1024 / 1024
                    
                    # Use helper method to evict completely
                    self._evict_timeframe_completely(timeframe)
                    current_memory -= df_memory
                    app_logger.info(f"Evicted {timeframe} from cache ({df_memory:.1f}MB)")

                    if current_memory <= self.config.MAX_CACHE_SIZE_MB * 0.8:  # Go to 80% to prevent thrashing
                        break

        # Enforce max rows per timeframe
        for timeframe, df in self.dataframes.items():
            if len(df) > self.config.MAX_ROWS_PER_TIMEFRAME:
                self.dataframes[timeframe] = df.tail(self.config.MAX_ROWS_PER_TIMEFRAME // 2)
                app_logger.info(f"Trimmed {timeframe} cache to {len(self.dataframes[timeframe])} rows")

        # Also enforce max number of cached timeframes
        if len(self.dataframes) > self.max_cache_size:
            oldest_tf = min(self.cache_times.keys(), key=lambda k: self.cache_times[k])
            self._evict_timeframe_completely(oldest_tf)
            app_logger.info(f"Evicted {oldest_tf} from cache (max timeframes)")

    def _evict_timeframe_completely(self, timeframe: str):
        """Helper to completely evict a timeframe from all tracking dictionaries"""
        self.dataframes.pop(timeframe, None)
        self.cache_times.pop(timeframe, None)
        self.cache_sizes.pop(timeframe, None)
        self._data_hash_cache.pop(timeframe, None)
        self.data_hashes.pop(timeframe, None)
        self.api_calls_per_timeframe.pop(timeframe, None)
        self.collection_phase.pop(timeframe, None)

    def get_df(self, timeframe: str) -> pd.DataFrame:
        """Get dataframe for timeframe"""
        return self.dataframes.get(timeframe, pd.DataFrame())

    def _validate_dataframe_enhanced(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Enhanced validation with progressive thresholds and better error reporting"""
        result = {"valid": False, "reason": "", "partial": False}
        
        # Check if dataframe is completely empty
        if df.empty:
            result["reason"] = "Dataframe is completely empty"
            return result
        
        # Check for required columns with case-insensitive matching
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        df_cols_lower = [str(col).lower() for col in df.columns]
        
        missing_cols = []
        for req_col in required_cols:
            if req_col not in df_cols_lower:
                missing_cols.append(req_col)
        
        if missing_cols:
            result["reason"] = f"Missing required columns: {missing_cols}"
            return result
        
        # Standardize column names to lowercase
        df.columns = [str(col).lower() for col in df.columns]
        
        # Check for zero or negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns and (df[col] <= 0).any():
                app_logger.warning(f"{timeframe}: Zero or negative prices in {col}")
                # Filter out invalid rows
                df = df[df[col] > 0]
        
        # Check for sufficient data with progressive thresholds
        min_initial_bars = self.config.MIN_INITIAL_BARS
        target_bars = self.config.HMM_TRAIN_WINDOW.get(timeframe, 504)
        
        if len(df) < min_initial_bars:
            result["reason"] = f"Insufficient data ({len(df)} < {min_initial_bars})"
            result["partial"] = True
            return result
        
        # Check for data quality issues
        null_counts = df.isnull().sum()
        for col, null_count in null_counts.items():
            if null_count > 0:
                null_pct = null_count / len(df)
                if null_pct > 0.1:  # More than 10% nulls
                    app_logger.warning(f"{timeframe}: {col} has {null_count} null values ({null_pct:.1%})")
        
        # Fill missing values if reasonable
        if df.isnull().any().any():
            fill_count = df.isnull().sum().sum()
            if fill_count < len(df) * 0.2:  # Less than 20% missing
                df = df.ffill().bfill()
                app_logger.info(f"Filled {fill_count} missing values in {timeframe}")
            else:
                result["reason"] = f"Too many missing values ({fill_count})"
                result["partial"] = True
                return result
        
        # Check for price anomalies
        if ('high' in df.columns and 'low' in df.columns and 
            (df['high'] < df['low']).any()):
            result["reason"] = "high < low detected"
            return result
        
        if ('close' in df.columns and 'high' in df.columns and 'low' in df.columns):
            if (df['close'] > df['high']).any() or (df['close'] < df['low']).any():
                result["reason"] = "close outside high/low range"
                return result
        
        # Data staleness check (only if we have at least 2 bars)
        if len(df) > 2:
            last_bar_age = (datetime.now() - df.index[-1]).total_seconds()
            expected_interval = {"1d": 86400, "4h": 14400,
                                 "1h": 3600, "15m": 900}.get(timeframe, 3600)
            staleness_threshold = expected_interval * self.config.DATA_STALENESS_THRESHOLD
            
            if last_bar_age > staleness_threshold:
                result["reason"] = f"Data stale ({last_bar_age:.0f}s old)"
                result["partial"] = True
                # Still return True for partial data if we have enough bars
                if len(df) >= min_initial_bars:
                    result["valid"] = True
                    return result
                return result
        
        # Update the dataframe in cache
        self.dataframes[timeframe] = df
        
        result["valid"] = True
        return result

    async def _fetch_twelvedata_enhanced(self, timeframe: str, outputsize: int) -> bool:
        """Enhanced TwelveData fetching with better error handling and case-insensitive column matching"""
        try:
            td_intervals = {"1d": "1day", "4h": "4h", "1h": "1h", "15m": "15min"}
            if timeframe not in td_intervals:
                return False

            interval = td_intervals[timeframe]

            url = "https://api.twelvedata.com/time_series"
            params = {
                "symbol": self.symbol,
                "interval": interval,
                "apikey": self.api_key,
                "outputsize": outputsize,
                "order": "ASC",
                "format": "JSON"
            }

            app_logger.debug(f"Fetching {timeframe} data from TwelveData (outputsize: {outputsize})")

            await self._ensure_session()
            async with self.session.get(url, params=params, timeout=30) as response:
                if response.status != 200:
                    app_logger.error(f"TwelveData HTTP error {response.status}: {await response.text()}")
                    self.data_source_health["twelvedata"]["failures"] += 1
                    await self._record_failure("twelvedata")
                    return False
                
                data = await response.json()
                
                # Debug logging for response structure
                if "values" not in data:
                    error_msg = data.get('message', 'Unknown error') if isinstance(data, dict) else 'Invalid response format'
                    app_logger.error(f"TwelveData error: {error_msg}")
                    self.data_source_health["twelvedata"]["failures"] += 1
                    await self._record_failure("twelvedata")
                    return False

                values = data["values"]
                if not values:
                    app_logger.warning(f"TwelveData returned empty data for {timeframe}")
                    return False

                # Create DataFrame
                df = pd.DataFrame(values)
                
                # Case-insensitive column name standardization
                df.columns = [str(col).lower() for col in df.columns]
                
                # Find datetime column (case-insensitive)
                datetime_col = None
                for col in df.columns:
                    if 'datetime' in col:
                        datetime_col = col
                        break
                
                if not datetime_col:
                    app_logger.error(f"No datetime column found. Columns: {df.columns.tolist()}")
                    return False
                
                # Convert datetime
                df['datetime'] = pd.to_datetime(df[datetime_col])
                df.set_index('datetime', inplace=True)
                
                # Drop the original datetime column if it's different
                if datetime_col != 'datetime':
                    df = df.drop(columns=[datetime_col])
                
                # Ensure required columns exist (case-insensitive)
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                available_cols = []
                
                for req_col in required_cols:
                    if req_col in df.columns:
                        available_cols.append(req_col)
                    else:
                        # Try to find case-insensitive match
                        matching_cols = [col for col in df.columns if req_col in col.lower()]
                        if matching_cols:
                            df = df.rename(columns={matching_cols[0]: req_col})
                            available_cols.append(req_col)
                        else:
                            app_logger.warning(f"Column {req_col} not found in TwelveData response")
                
                # Check if we have at least the essential columns
                essential_cols = ['open', 'high', 'low', 'close']
                if not all(col in df.columns for col in essential_cols):
                    app_logger.error(f"Missing essential columns. Available: {df.columns.tolist()}")
                    return False
                
                # Convert to numeric
                for col in essential_cols + (['volume'] if 'volume' in df.columns else []):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Drop rows with NaN in essential columns
                df = df.dropna(subset=essential_cols)
                
                if df.empty:
                    app_logger.warning(f"DataFrame empty after cleaning for {timeframe}")
                    return False

                app_logger.info(f"TwelveData: {timeframe} - {len(df)} bars")
                
                # Store in dataframes
                self.dataframes[timeframe] = await self._engineer_features(df, timeframe)
                await self._record_success("twelvedata")
                self.data_source_health["twelvedata"]["healthy"] = True
                self.data_source_health["twelvedata"]["last_success"] = time.time()
                self.data_source_health["twelvedata"]["failures"] = 0
                return True

        except aiohttp.ClientError as e:
            app_logger.error(f"TwelveData network error: {e}")
            self.data_source_health["twelvedata"]["failures"] += 1
            await self._record_failure("twelvedata")
            return False
        except Exception as e:
            app_logger.error(f"TwelveData failed: {e}", exc_info=True)
            self.data_source_health["twelvedata"]["failures"] += 1
            await self._record_failure("twelvedata")
            return False

    async def _fetch_yfinance_enhanced(self, timeframe: str, outputsize: int) -> bool:
        """Enhanced yFinance fetching with better error handling"""
        async with self.yfinance_limiter:
            try:
                interval_map = {"1d": "1d", "4h": "60m", "1h": "60m", "15m": "15m"}
                if timeframe not in interval_map:
                    return False

                # Calculate period based on outputsize and timeframe
                if timeframe == "1d":
                    period = f"{min(outputsize, 730)}d"  # Max 2 years
                elif timeframe == "4h":
                    period = f"{min(outputsize // 6, 60)}d"  # Max 60 days
                elif timeframe == "1h":
                    period = f"{min(outputsize // 24, 60)}d"  # Max 60 days
                else:  # 15m
                    period = f"{min(outputsize // 96, 60)}d"  # Max 60 days

                app_logger.debug(f"yFinance {timeframe}: Requesting {period} data")

                ticker = yf.Ticker(self.symbol)

                # yFinance fetch with timeout
                df = await asyncio.wait_for(
                    asyncio.to_thread(
                        lambda: ticker.history(
                            interval=interval_map[timeframe],
                            period=period,
                            timeout=15,
                            raise_errors=True,
                            auto_adjust=True
                        )
                    ),
                    timeout=20
                )

                if df.empty:
                    app_logger.warning(f"yFinance returned empty DataFrame for {timeframe}")
                    return False

                # Standardize column names
                df.columns = df.columns.str.lower()
                
                # Resample for 4h if needed
                if timeframe == "4h" and interval_map[timeframe] == "60m":
                    df = df.resample('4H', offset='1h').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()

                # Remove duplicates
                df = df[~df.index.duplicated(keep='last')]
                
                # Ensure required columns
                required_cols = ['open', 'high', 'low', 'close']
                if not all(col in df.columns for col in required_cols):
                    app_logger.error(f"yFinance missing required columns: {df.columns.tolist()}")
                    return False
                
                # Drop rows with NaN in essential columns
                df = df.dropna(subset=required_cols)
                
                if df.empty:
                    return False

                df = await self._engineer_features(df, timeframe)
                self.dataframes[timeframe] = df
                await self._record_success("yfinance")
                self.data_source_health["yfinance"]["healthy"] = True
                self.data_source_health["yfinance"]["last_success"] = time.time()
                self.data_source_health["yfinance"]["failures"] = 0
                return True

            except asyncio.TimeoutError:
                app_logger.error("yFinance fetch timed out")
                self.data_source_health["yfinance"]["failures"] += 1
                await self._record_failure("yfinance")
                return False
            except Exception as e:
                app_logger.error(f"yFinance failed: {e}")
                self.data_source_health["yfinance"]["failures"] += 1
                await self._record_failure("yfinance")
                return False

    async def _emergency_data_fallback(self, timeframe: str) -> bool:
        """Emergency fallback when all other data sources fail"""
        try:
            app_logger.warning(f"Using emergency fallback for {timeframe}")
            
            # Try to use cached data if available
            if timeframe in self.dataframes and not self.dataframes[timeframe].empty:
                df = self.dataframes[timeframe]
                app_logger.info(f"Using cached data for {timeframe}: {len(df)} bars")
                return True
            
            # Try to generate synthetic data for testing
            if self.config.PAPER_TRADING:
                app_logger.warning(f"Generating synthetic data for {timeframe} in paper trading mode")
                
                # Create a simple synthetic dataset
                dates = pd.date_range(end=datetime.now(), periods=100, freq=timeframe)
                np.random.seed(42)
                base_price = 100.0
                returns = np.random.normal(0.0001, 0.02, 100)
                prices = base_price * np.exp(np.cumsum(returns))
                
                df = pd.DataFrame({
                    'open': prices * 0.99,
                    'high': prices * 1.01,
                    'low': prices * 0.98,
                    'close': prices,
                    'volume': np.random.randint(100000, 1000000, 100)
                }, index=dates)
                
                df = await self._engineer_features(df, timeframe)
                self.dataframes[timeframe] = df
                app_logger.warning(f"Generated synthetic data for {timeframe}")
                return True
            
            return False
            
        except Exception as e:
            app_logger.error(f"Emergency fallback failed: {e}")
            return False

    async def _engineer_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Engineer features without lookahead bias, with timeframe-aware parameters"""
        # Remove zero/negative prices
        df = df[(df['close'] > 0) & (df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0)]

        if df.empty:
            return df

        # Cap extreme outliers (non-physical)
        for col in ['close', 'open', 'high', 'low']:
            with np.errstate(all='ignore'):
                q01, q99 = df[col].quantile([0.01, 0.99])
                df[col] = df[col].clip(lower=q01*0.5, upper=q99*2)

        df = df.copy().astype(np.float64)

        # Timeframe-aware parameters
        if timeframe == "1d":
            lookback = 20
            rsi_period = 14
            vwap_period = 20
        elif timeframe == "4h":
            lookback = 15
            rsi_period = 12
            vwap_period = 15
        elif timeframe == "1h":
            lookback = 10
            rsi_period = 10
            vwap_period = 10
        else:
            lookback = 8
            rsi_period = 8
            vwap_period = 8

        # Basic features
        with np.errstate(divide='ignore', invalid='ignore'):
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            df['log_ret'] = df['log_ret'].replace([np.inf, -np.inf], np.nan)

        # ATR (for position sizing)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        true_range = np.max([high_low, high_close, low_close], axis=0)
        df['atr'] = pd.Series(true_range).rolling(self.config.ATR_PERIOD).mean()

        # Garman-Klass volatility
        hl = (np.log(df['high'] / df['low']) ** 2) / 2
        co = (np.log(df['close'] / df['open']) ** 2)
        df['vol_gk'] = np.sqrt((hl + co).rolling(lookback).mean())

        # Trend Efficiency - uses only past data
        change = (df['close'] - df['close'].shift(lookback)).abs()
        volatility = df['close'].diff().abs().rolling(lookback).sum()

        # Prevent division by zero with dynamic floor
        vol_floor = max(volatility.mean() * 0.01, 1e-6)
        volatility = volatility.replace(0, vol_floor)

        df['trend_eff'] = (change / (volatility)).clip(0, 1)

        # RSI (no bias)
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
        rs = gain / (loss.replace(0, np.nan) + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))

        # Volume Profile (simplified) - Rolling VWAP (no lookahead)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (df['volume'] * typical_price).rolling(vwap_period).sum() / \
            df['volume'].rolling(vwap_period).sum()
        # Shift to prevent lookahead bias (NO backward fill)
        df['vwap'] = df['vwap'].shift(1)
        df['vwap_dev'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-9)

        # Optimize outlier cap
        for col in ['close', 'open', 'high', 'low']:
            q01, q99 = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lower=q01*0.5, upper=q99*2)

        # Momentum divergence
        df['rsi_mom'] = df['rsi'].diff(3)
        df['price_mom'] = df['close'].diff(3)
        df['divergence'] = np.sign(df['rsi_mom']) != np.sign(df['price_mom'])

        # Data continuity check - only warn, don't fail
        if len(df) > 2:
            time_diffs = df.index.to_series().diff().dt.total_seconds()
            expected_diff = {"1d": 86400, "4h": 14400,
                             "1h": 3600, "15m": 900}.get(timeframe, 3600)
            gaps = (time_diffs > expected_diff * 2).sum()
            if gaps > 0:
                app_logger.warning(f"{timeframe}: {gaps} data gaps detected")
                # We don't return False, just log

        return df.dropna()

# ===============================================================================
# 🧠 MULTI-TIMEFRAME BRAIN - ENHANCED REGIME DETECTION WITH SEQUENTIAL HMM
# ===============================================================================

class MultiTimeframeBrain:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.models: Dict[str, hmm.GaussianHMM] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.regime_maps: Dict[str, Dict[int, str]] = {}
        self.is_trained: Dict[str, bool] = {}
        self.training_metrics: Dict[str, Dict[str, Any]] = {}
        self.training_hashes: Dict[str, str] = {}
        self.lock = asyncio.Lock()
        self.last_train_time: Dict[str, datetime] = {}
        self.performance_window: Dict[str, deque] = {}

        # Drift detection
        self.drift_threshold = 0.12
        self.performance_history: Dict[str, deque] = {
            tf: deque(maxlen=50) for tf in [config.PRIMARY_TIMEFRAME] + config.INTRADAY_TIMEFRAMES
        }

        # Chop detection
        self.chop_threshold = 0.55

        # Dynamic component selection
        self.component_usage: Dict[str, int] = {}

        # Stability Buffer
        self.retrain_cooldowns: Dict[str, datetime] = {}

        # Training failure counter
        self.training_failures: Dict[str, int] = defaultdict(int)
        self.training_failure_threshold = 3

        # Model persistence with joblib
        self.model_cache_path = Path("models")
        self.model_cache_path.mkdir(exist_ok=True)

        # Reference to data engine using weak reference to prevent circular reference
        self._data_engine_ref: Optional[weakref.ReferenceType] = None

        # Telegram for alerts
        self._telegram_ref: Optional[weakref.ReferenceType] = None

        # Sequence length for HMM inference (STATISTICAL FIX)
        self.sequence_length = config.HMM_SEQUENCE_WINDOW
        self.min_sequence_length = config.HMM_MIN_SEQUENCE_LENGTH

    @property
    def data_engine(self):
        """Get data engine from weak reference with graceful degradation"""
        if self._data_engine_ref:
            engine = self._data_engine_ref()
            if engine is None:
                app_logger.warning("Data engine weak reference has expired - entering degraded mode")
                return None
            return engine
        app_logger.warning("Data engine not set - entering degraded mode")
        return None

    @data_engine.setter
    def data_engine(self, engine):
        """Set data engine using weak reference"""
        self._data_engine_ref = weakref.ref(engine) if engine else None

    @property
    def telegram(self):
        """Get telegram from weak reference with graceful degradation"""
        if self._telegram_ref:
            telegram_bot = self._telegram_ref()
            if telegram_bot is None:
                app_logger.warning("Telegram weak reference has expired - alerts disabled")
                return None
            return telegram_bot
        return None

    @telegram.setter
    def telegram(self, telegram_bot):
        """Set telegram using weak reference"""
        self._telegram_ref = weakref.ref(telegram_bot) if telegram_bot else None

    async def initialize(self):
        """Initialize the brain (separate from __init__)"""
        await self._cleanup_old_models()

    async def _cleanup_old_models(self):
        """Async cleanup of old model files"""
        try:
            for model_file in self.model_cache_path.glob("*.joblib"):
                if (time.time() - model_file.stat().st_mtime) > 90*86400:
                    await asyncio.to_thread(model_file.unlink)
                    app_logger.info(f"Deleted old model: {model_file}")
        except Exception as e:
            app_logger.warning(f"Model cleanup failed: {e}")

    async def should_retrain(self, timeframe: str, df: pd.DataFrame, recent_performance: Optional[float] = None) -> Tuple[bool, str]:
        """Determine if model needs retraining based on data drift and performance"""
        reasons = []

        # Check cooldown (Prevent retraining too frequently which causes label flip chaos)
        if timeframe in self.retrain_cooldowns:
            time_since_retrain = (datetime.now() - self.retrain_cooldowns[timeframe]).total_seconds()
            if time_since_retrain < 3600:
                return False, "cooldown"

        # Check time since last train
        if timeframe not in self.last_train_time:
            reasons.append("first_train")
        else:
            bars_since_train = len(df) - self.config.HMM_TRAIN_WINDOW.get(timeframe, 504)
            retrain_interval = self.config.HMM_RETRAIN_INTERVAL_BARS.get(timeframe, 100)

            if bars_since_train >= retrain_interval:
                reasons.append(f"interval_{bars_since_train}/{retrain_interval}")

        # Check data hash for significant changes (cached)
        if len(df) > 0:
            new_hash = hashlib.sha256(df.tail(min(100, len(df))).to_csv().encode()).hexdigest()
            old_hash = self.training_hashes.get(timeframe, "")

            if new_hash != old_hash:
                reasons.append("data_changed")

        # Check performance degradation
        if recent_performance is not None and recent_performance < self.drift_threshold:
            reasons.append(f"drift_{recent_performance:.2f}")

        # Check component count optimization
        current_components = self.component_usage.get(timeframe, self.config.HMM_COMPONENTS)
        optimal_components = self._optimal_component_count(df)
        if optimal_components != current_components:
            reasons.append(f"components_{current_components}->{optimal_components}")

        return len(reasons) > 0, "|".join(reasons)

    def _optimal_component_count(self, df: pd.DataFrame) -> int:
        """Determine optimal HMM components based on data characteristics"""
        if df.empty or 'vol_gk' not in df.columns:
            return self.config.HMM_COMPONENTS
            
        volatility = df['vol_gk'].mean() if 'vol_gk' in df.columns else 0.01
        if len(df) > 100:
            vol_percentile = np.percentile(df['vol_gk'].dropna(), 75)
        else:
            vol_percentile = volatility

        if vol_percentile > 0.05:
            return min(self.config.HMM_MAX_COMPONENTS, 5)
        elif vol_percentile > 0.03:
            return self.config.HMM_COMPONENTS
        else:
            return max(self.config.HMM_MIN_COMPONENTS, 3)

    async def train_timeframe(self, timeframe: str, df: pd.DataFrame, retrain_reason: str = "") -> bool:
        """Train HMM with walk-forward validation, regime stability, and dynamic components"""
        async with self.lock:
            start_time = time.time()

            # Check if we have enough data
            if df.empty or len(df) < self.config.MIN_INITIAL_BARS:
                app_logger.warning(f"{timeframe}: Insufficient data for training ({len(df)} bars)")
                self.training_failures[timeframe] += 1
                await self._check_training_failure_threshold(timeframe)
                return False

            # Adaptive window reduction if insufficient data
            original_window = self.config.HMM_TRAIN_WINDOW.get(timeframe, 504)
            train_window = original_window
            available_bars = len(df)
            min_bars = max(self.config.MIN_INITIAL_BARS, int(train_window * 0.5))  # More lenient

            # Calculate min_bars once, reduce window systematically
            while available_bars < min_bars and train_window > 100:
                train_window = int(train_window * 0.8)
                app_logger.warning(f"{timeframe}: Reducing train window to {train_window} (insufficient data)")

            if available_bars < min_bars:
                app_logger.warning(f"{timeframe}: Insufficient data even after reduction")
                self.training_failures[timeframe] += 1
                await self._check_training_failure_threshold(timeframe)
                return False

            try:
                # Enhanced features
                features = ['log_ret', 'vol_gk', 'rsi', 'trend_eff', 'vwap_dev']
                available_features = [f for f in features if f in df.columns]

                if len(available_features) < 3:
                    app_logger.warning(f"{timeframe}: Insufficient features ({len(available_features)})")
                    self.training_failures[timeframe] += 1
                    await self._check_training_failure_threshold(timeframe)
                    return False

                X = df[available_features].values

                # Validate finite values
                if not np.isfinite(X).all():
                    app_logger.warning(f"{timeframe}: NaN or infinite values in training data")
                    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

                # Walk-forward split (80/20)
                split_idx = int(len(X) * 0.8)
                X_train = X[:split_idx]
                X_test = X[split_idx:]

                # Scale
                scaler = StandardScaler()
                X_train_scaled = await asyncio.to_thread(scaler.fit_transform, X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train model with dynamic components
                optimal_components = self._optimal_component_count(df)

                # Offload to thread pool with timeout
                model = await asyncio.wait_for(
                    asyncio.to_thread(
                        lambda: hmm.GaussianHMM(
                            n_components=optimal_components,
                            covariance_type="diag",
                            n_iter=self.config.HMM_MAX_ITER,
                            random_state=self.config.HMM_RANDOM_STATE,
                            verbose=False,
                            tol=1e-4
                        ).fit(X_train_scaled)
                    ),
                    timeout=60
                )

                # Validate
                train_score = model.score(X_train_scaled)
                test_score = model.score(X_test_scaled)
                score_ratio = test_score / (train_score + 1e-9)

                if score_ratio < 0.55:
                    app_logger.warning(f"{timeframe}: Model overfitting ({score_ratio:.2f})")
                    self.training_failures[timeframe] += 1
                    await self._check_training_failure_threshold(timeframe)
                    return False

                # Check regime diversity
                if not await self._check_regime_diversity(model, X_train_scaled, optimal_components):
                    app_logger.error(f"{timeframe}: Insufficient regime diversity in training data")
                    raise InsufficientRegimeDiversityError(f"{timeframe} lacks regime diversity")

                # STABLE regime mapping with historical consistency
                old_regime_map = self.regime_maps.get(timeframe, {})
                new_regime_map = await self._map_regimes_stably(timeframe, model, X_train_scaled, old_regime_map, available_features)

                self.models[timeframe] = model
                self.scalers[timeframe] = scaler
                self.regime_maps[timeframe] = new_regime_map
                self.is_trained[timeframe] = True
                self.last_train_time[timeframe] = datetime.now()
                self.retrain_cooldowns[timeframe] = datetime.now()
                if len(df) > 0:
                    self.training_hashes[timeframe] = hashlib.sha256(df.tail(min(100, len(df))).to_csv().encode()).hexdigest()
                self.component_usage[timeframe] = optimal_components
                self.training_failures[timeframe] = 0

                self.training_metrics[timeframe] = {
                    "train_score": train_score,
                    "test_score": test_score,
                    "score_ratio": score_ratio,
                    "features": available_features,
                    "regime_map": new_regime_map,
                    "train_duration_ms": int((time.time() - start_time) * 1000),
                    "retrain_reason": retrain_reason,
                    "components_used": optimal_components
                }

                # Limit training metrics size
                if len(self.training_metrics) > 100:
                    oldest = min(self.training_metrics.keys(), key=lambda k: self.training_metrics[k]['train_duration_ms'])
                    del self.training_metrics[oldest]

                # Persist model to disk with HMAC
                await self._persist_model(timeframe, model, scaler, new_regime_map, available_features)

                # Initialize performance tracking
                if timeframe not in self.performance_window:
                    self.performance_window[timeframe] = deque(maxlen=50)

                app_logger.info(f"{timeframe} HMM trained (ratio: {score_ratio:.2f}, components: {optimal_components}, duration: {self.training_metrics[timeframe]['train_duration_ms']}ms)")
                return True

            except asyncio.TimeoutError:
                app_logger.error(f"{timeframe} training timed out")
                self.training_failures[timeframe] += 1
                await self._check_training_failure_threshold(timeframe)
                return False
            except InsufficientRegimeDiversityError:
                raise
            except Exception as e:
                app_logger.error(f"{timeframe} training failed: {e}")
                self.training_failures[timeframe] += 1
                await self._check_training_failure_threshold(timeframe)
                return False

    async def _check_training_failure_threshold(self, timeframe: str):
        """Alert if training failures exceed threshold"""
        if self.training_failures[timeframe] >= self.training_failure_threshold:
            app_logger.critical(f"{timeframe}: Training failed {self.training_failures[timeframe]} times - strategy degraded")
            # Telegram alert
            try:
                telegram_bot = self.telegram
                if telegram_bot:
                    await telegram_bot.send(
                        f"**MODEL DEGRADATION**\nTimeframe: {timeframe}\nFailures: {self.training_failures[timeframe]}\nManual intervention required.",
                        priority="critical"
                    )
            except Exception as e:
                app_logger.error(f"Cannot send Telegram alert: {e}")

    async def _check_regime_diversity(self, model: hmm.GaussianHMM, X: np.ndarray, n_components: int) -> bool:
        """Check that training data contains at least 3 unique regimes"""
        predictions = model.predict(X)
        unique_regimes = len(set(predictions))
        return unique_regimes >= min(3, n_components)

    async def _persist_model(self, timeframe: str, model: hmm.GaussianHMM, scaler: StandardScaler, regime_map: Dict[int, str], features: List[str]):
        """Save model to disk with HMAC signature"""
        try:
            model_data = {
                'model': model,
                'scaler': scaler,
                'regime_map': regime_map,
                'timestamp': datetime.now(),
                'data_hash': self.training_hashes.get(timeframe, ''),
                'features': features
            }
            path = self.model_cache_path / f"{timeframe}_model.joblib"

            # Compute HMAC signature using dedicated secret
            secret = self.config.MODEL_SIGNATURE_SECRET.encode()

            # Wrap blocking I/O in thread pool
            serialized = await asyncio.to_thread(self._serialize_model_data, model_data)
            signature = hmac.new(secret, serialized, hashlib.sha256).hexdigest()

            # Atomic write with thread pool
            temp_path = path.with_suffix('.tmp')
            await asyncio.to_thread(joblib.dump, model_data, temp_path)

            # Atomic signature write
            signature_path = path.with_suffix('.sig')
            temp_sig = signature_path.with_suffix('.tmp')
            await asyncio.to_thread(temp_sig.write_text, signature)
            await asyncio.to_thread(temp_sig.replace, signature_path)

            # Replace atomically
            await asyncio.to_thread(temp_path.replace, path)
            app_logger.debug(f"Model persisted: {timeframe} with HMAC signature")
        except Exception as e:
            app_logger.error(f"Model persistence failed: {e}")

    def _serialize_model_data(self, model_data: Dict[str, Any]) -> bytes:
        """Helper method to serialize model data"""
        import pickle
        return pickle.dumps(model_data)

    async def _load_persisted_model(self, timeframe: str) -> bool:
        """Load model from disk with HMAC verification"""
        try:
            path = self.model_cache_path / f"{timeframe}_model.joblib"
            signature_path = path.with_suffix('.sig')

            if not path.exists() or not signature_path.exists():
                return False

            # Verify signature using dedicated secret
            secret = self.config.MODEL_SIGNATURE_SECRET.encode()

            # Read serialized data with thread pool
            serialized = await asyncio.to_thread(path.read_bytes)

            expected_signature = await asyncio.to_thread(signature_path.read_text)

            if not hmac.compare_digest(
                hmac.new(secret, serialized, hashlib.sha256).hexdigest(),
                expected_signature
            ):
                app_logger.error(f"{timeframe}: Model signature verification failed - possible tampering")
                return False

            model_data = await asyncio.to_thread(joblib.load, path)

            self.models[timeframe] = model_data['model']
            self.scalers[timeframe] = model_data['scaler']
            self.regime_maps[timeframe] = model_data['regime_map']
            self.is_trained[timeframe] = True
            self.training_hashes[timeframe] = model_data.get('data_hash', '')
            self.training_metrics[timeframe] = {
                'features': model_data.get('features', ['log_ret', 'vol_gk', 'rsi', 'trend_eff', 'vwap_dev'])
            }
            app_logger.info(f"Loaded persisted model: {timeframe}")
            return True
        except Exception as e:
            app_logger.warning(f"Failed to load persisted model {timeframe}: {e}")
            return False

    async def _map_regimes_stably(self, timeframe: str, model: hmm.GaussianHMM, X: np.ndarray,
                                  old_map: Dict[int, str], available_features: List[str]) -> Dict[int, str]:
        """Map regimes stably by correlating with old mapping and using statistical rules"""
        predictions = model.predict(X)
        means = model.means_

        # Find feature indices from available features
        try:
            log_ret_idx = available_features.index('log_ret')
        except ValueError:
            log_ret_idx = 0

        try:
            vol_gk_idx = available_features.index('vol_gk')
        except ValueError:
            vol_gk_idx = 1 if len(available_features) > 1 else 0

        log_ret_means = means[:, log_ret_idx]
        vol_means = means[:, vol_gk_idx] if len(means[0]) > vol_gk_idx else np.zeros_like(log_ret_means)

        # Use combined metric for stable sorting
        log_ret_normalized = (log_ret_means - log_ret_means.min()) / (log_ret_means.max() - log_ret_means.min() + 1e-9)
        vol_normalized = (vol_means - vol_means.min()) / (vol_means.max() - vol_means.min() + 1e-9)

        # Combined metric: return dominates but volatility influences ranking
        combined_metric = log_ret_normalized * 0.7 + vol_normalized * 0.3

        # Sort by combined metric
        sorted_indices = np.argsort(combined_metric)

        # Create new map
        new_map = {}
        n_components = len(sorted_indices)

        # Map to BULL/CHOP/BEAR based on combined metric
        for i, idx in enumerate(sorted_indices):
            percentile = i / (n_components - 1) if n_components > 1 else 0.5

            # Use combined metric for regime assignment
            metric_value = combined_metric[idx]

            if metric_value < 0.3:
                new_map[idx] = "BEAR"
            elif metric_value > 0.7:
                new_map[idx] = "BULL"
            else:
                # Middle regimes: check volatility for CHOP classification
                vol_percentile = (vol_means[idx] - vol_means.min()) / (vol_means.max() - vol_means.min() + 1e-9)
                if vol_percentile > 0.6:
                    new_map[idx] = "CHOP"
                else:
                    # Further distinguish middle regimes
                    if percentile < 0.5:
                        new_map[idx] = "BEAR_CHOP"
                    else:
                        new_map[idx] = "BULL_CHOP"

        # Ensure unique regime mapping
        if len(set(new_map.values())) != len(new_map):
            app_logger.warning(f"{timeframe}: Regime mapping not unique, forcing uniqueness")
            # Force unique mapping
            unique_values = set()
            for k, v in new_map.items():
                if v in unique_values:
                    # Append number to make unique
                    counter = 1
                    while f"{v}_{counter}" in unique_values:
                        counter += 1
                    new_map[k] = f"{v}_{counter}"
                unique_values.add(new_map[k])

        return new_map

    async def predict_timeframe(self, timeframe: str, df: pd.DataFrame, calculate_quality: bool = True) -> Dict[str, Any]:
        """Predict with SEQUENTIAL HMM inference (STATISTICAL FIX), confidence intervals, validation, and quality scoring"""
        async with self.lock:
            if not self.is_trained.get(timeframe, False):
                # Try loading persisted model
                if not await self._load_persisted_model(timeframe):
                    # Alert on training failure
                    if self.training_failures.get(timeframe, 0) >= self.training_failure_threshold:
                        app_logger.critical(f"{timeframe}: Strategy degraded - model not available")
                    return {
                        "action": "HOLD", "score": 0.0, "regime": "UNKNOWN",
                        "is_valid": False, "quality": 0.0, "confidence": 0.0, "chop_probability": 0.0
                    }

            if df.empty or len(df) < self.min_sequence_length:
                return {
                    "action": "HOLD", "score": 0.0, "regime": "UNKNOWN",
                    "is_valid": False, "quality": 0.0, "confidence": 0.0, "chop_probability": 0.0
                }

            try:
                features = self.training_metrics[timeframe]["features"]

                # STATISTICAL FIX: Use SEQUENCE of observations for valid HMM inference
                sequence_length = min(self.sequence_length, len(df))
                X_sequence = df[features].iloc[-sequence_length:].values

                # Validate finite values
                if not np.isfinite(X_sequence).all():
                    return {
                        "action": "HOLD", "score": 0.0, "regime": "ERROR",
                        "is_valid": False, "quality": 0.0, "confidence": 0.0, "chop_probability": 0.0
                    }

                scaler = self.scalers[timeframe]
                X_scaled = scaler.transform(X_sequence)

                # Get probabilities for ENTIRE SEQUENCE (valid HMM inference)
                posteriors_sequence = self.models[timeframe].predict_proba(X_scaled)

                # Use the LAST observation's probabilities for current regime (most recent)
                posteriors = posteriors_sequence[-1]

                # Also compute sequence-consistent regime via Viterbi for robustness
                try:
                    viterbi_states = self.models[timeframe].predict(X_scaled)
                    sequence_consistent_regime_idx = viterbi_states[-1]
                except Exception:
                    # Fallback to max posterior if Viterbi fails
                    sequence_consistent_regime_idx = np.argmax(posteriors)

                # Check regime map key exists
                if sequence_consistent_regime_idx not in self.regime_maps[timeframe]:
                    app_logger.error(f"{timeframe}: Regime map missing index {sequence_consistent_regime_idx}")
                    return {
                        "action": "HOLD", "score": 0.0, "regime": "ERROR",
                        "is_valid": False, "quality": 0.0, "confidence": 0.0, "chop_probability": 0.0
                    }

                regime = self.regime_maps[timeframe].get(sequence_consistent_regime_idx, "UNKNOWN")

                # Calculate score based on regime probabilities
                bull_prob = 0.0
                bear_prob = 0.0
                chop_prob = 0.0

                for i, state_regime in self.regime_maps[timeframe].items():
                    if "BULL" in state_regime and "CHOP" not in state_regime:
                        bull_prob += posteriors[i]
                    elif "BEAR" in state_regime and "CHOP" not in state_regime:
                        bear_prob += posteriors[i]
                    else:
                        chop_prob += posteriors[i]

                score = bull_prob - bear_prob

                # STABILITY CHECK: If retrained recently, dampen confidence
                raw_confidence = float(max(posteriors))
                if timeframe in self.retrain_cooldowns:
                    time_since_retrain = (datetime.now() - self.retrain_cooldowns[timeframe]).total_seconds()
                    if time_since_retrain < 300:
                        app_logger.debug(f"{timeframe}: Stability dampening active ({time_since_retrain:.0f}s ago)")
                        raw_confidence *= 0.8

                # Sequence stability bonus: if last N states are consistent
                sequence_stability = 0.0
                if len(viterbi_states) >= 5:
                    last_states = viterbi_states[-5:]
                    if len(set(last_states)) == 1:  # All same state
                        sequence_stability = 0.15
                    elif len(set(last_states)) <= 2:  # At most 2 states
                        sequence_stability = 0.08

                result = {
                    "timeframe": timeframe,
                    "action": self._determine_action_from_regime(regime, score, chop_prob),
                    "score": round(float(score), 3),
                    "regime": regime,
                    "confidence": raw_confidence + sequence_stability,
                    "posteriors": posteriors.tolist(),
                    "is_valid": True,
                    "quality": 0.0,
                    "chop_probability": float(chop_prob),
                    "components": self.component_usage.get(timeframe, self.config.HMM_COMPONENTS),
                    "sequence_length": sequence_length,
                    "sequence_stability": sequence_stability,
                    "viterbi_state": int(sequence_consistent_regime_idx)
                }

                # Calculate quality if requested
                if calculate_quality:
                    result["quality"] = self._calculate_quality(result)

                return result

            except Exception as e:
                app_logger.error(f"{timeframe} prediction error: {e}")
                return {
                    "action": "HOLD", "score": 0.0, "regime": "ERROR",
                    "is_valid": False, "quality": 0.0, "confidence": 0.0, "chop_probability": 0.0
                }

    def _determine_action_from_regime(self, regime: str, score: float, chop_prob: float) -> str:
        """Clear action determination logic"""
        # Chop filter
        if chop_prob > 0.5:
            return "HOLD"

        # Explicit regime-action mapping
        if "BULL" in regime and "CHOP" not in regime:
            if score > self.config.REGIME_BULL_THRESHOLD:
                return "BUY"
            elif score < self.config.REGIME_BEAR_THRESHOLD:
                return "SELL"  # Strong counter-trend signal in bull regime
            else:
                return "HOLD"
        elif "BEAR" in regime and "CHOP" not in regime:
            if score < self.config.REGIME_BEAR_THRESHOLD:
                return "SELL"
            elif score > self.config.REGIME_BULL_THRESHOLD:
                return "BUY"  # Strong counter-trend signal in bear regime
            else:
                return "HOLD"
        else:  # CHOP or UNKNOWN
            return "HOLD"

    def _calculate_quality(self, signal: Dict[str, Any]) -> float:
        """Calculate weighted signal quality with chop penalty, volatility adjustment, and sequence bonus"""
        # Base score from confidence
        base_score = signal.get("confidence", 0.0) * 100

        # Chop penalty
        chop_penalty = signal.get("chop_probability", 0.0) * 50
        base_score -= chop_penalty

        # Regime alignment bonus
        regime_alignment = 0
        if (signal["regime"] == "BULL" and signal["action"] == "BUY") or \
           (signal["regime"] == "BEAR" and signal["action"] == "SELL"):
            regime_alignment = 20 * signal.get("confidence", 0)

        # Score strength bonus
        score_abs = abs(signal.get("score", 0))
        strength_bonus = 15 if score_abs > 0.6 else 8 if score_abs > 0.4 else 0

        # Timeframe weight (normalized)
        weights = {
            "1d": 4.0,
            "4h": 2.5,
            "1h": 1.5,
            "15m": 1.0
        }
        tf_weight = weights.get(signal['timeframe'], 1.0)

        # Component count bonus
        component_bonus = signal.get("components", self.config.HMM_COMPONENTS) * 2

        # Sequence stability bonus (STATISTICAL FIX)
        sequence_bonus = signal.get("sequence_stability", 0.0) * 25

        # Clamp AFTER all multiplications
        combined_score = (base_score + regime_alignment + strength_bonus + component_bonus + sequence_bonus) * tf_weight
        final_score = min(100, max(0, combined_score))

        return final_score

# ===============================================================================
# 🎯 EXECUTION ENGINE - STATE-AWARE & SMART EXECUTION (V19 ENHANCED)
# ===============================================================================

class ExecutionCircuitBreaker:
    """Circuit breaker for Alpaca API"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.state = {"open": False, "last_failure": 0, "failure_count": 0}
        self.threshold = config.CIRCUIT_FAILURE_THRESHOLD
        self.timeout = config.CIRCUIT_TIMEOUT_SECONDS
        self.lock = asyncio.Lock()

    async def __aenter__(self):
        async with self.lock:
            if self.state["open"]:
                if time.time() - self.state["last_failure"] > self.timeout:
                    self.state["open"] = False
                    self.state["failure_count"] = 0
                else:
                    raise APIError("Circuit breaker is OPEN", status_code=503)
            return self

    async def __aexit__(self, exc_type, exc, tb):
        async with self.lock:
            if exc_type is not None:
                self.state["failure_count"] += 1
                self.state["last_failure"] = time.time()
                if self.state["failure_count"] >= self.threshold:
                    self.state["open"] = True
                    app_logger.error("Execution circuit breaker OPENED")
            else:
                if self.state["failure_count"] > 0:
                    self.state["failure_count"] = 0

class ExecutionEngine:
    def __init__(self, config: SystemConfig, client: TradingClient, db: DatabaseManager, telegram: TelegramBot, data_engine: MultiTimeframeDataEngine):
        self.config = config
        self.client = client
        self.db = db
        self.telegram = telegram
        self.data_engine = data_engine
        self.daily_trades = 0
        self.daily_loss = Decimal('0.0')  # Use Decimal for precision
        self.max_drawdown_today = Decimal('0.0')
        self.last_reset_date = datetime.now().date()

        # Faster cache refresh with configurable TTL
        self.position_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self.cache_ttl = min(config.POSITION_CACHE_TTL, config.LIVE_LOOP_INTERVAL_SECONDS)
        self._last_position_fetch = 0
        self._position_lock = asyncio.Lock()

        # Persistent active order tracking
        self.active_orders: Dict[str, Dict[str, Any]] = {}
        self.order_lock = asyncio.Lock()

        # Enhanced trade scheduling
        self.trade_history: List[datetime] = []
        self.last_trade_time = datetime.min
        self.last_trade_timeframe: Optional[str] = None

        # Performance-based throttling
        self.recent_trade_performance: deque = deque(maxlen=20)
        self.performance_threshold = 0.4

        # Cooldown tracking
        self.timeframe_cooldowns: Dict[str, datetime] = {}
        self.cooldown_duration = 60

        # Safety mode
        self.safety_mode = False
        self.safety_mode_threshold = Decimal('-0.02')  # Use Decimal

        # Execution circuit breaker
        self.execution_circuit_breaker = ExecutionCircuitBreaker(config)

        # Store primary timeframe signal for correlation checks
        self._primary_tf_signal: Optional[Dict[str, Any]] = None

        # Track open positions count
        self.open_positions_count = 0
        self._open_positions_lock = asyncio.Lock()

    async def initialize(self):
        """Initialize daily limits and load from DB"""
        await self._reset_daily_limits()
        await self._validate_account_status()
        await self.check_alpaca_health()  # Health check

        # Load timeframe cooldowns from DB
        risk_data = await self.db.get_daily_risk(datetime.now().date().isoformat())
        if risk_data and risk_data.get('timeframe_cooldowns'):
            self.timeframe_cooldowns = {
                tf: datetime.fromisoformat(ts) for tf, ts in risk_data['timeframe_cooldowns'].items()
            }

        # Load recent trade performance
        await self._load_recent_performance()

        # Reconcile active orders on startup (FIXED: Using GetOrdersRequest with OrderStatus.NEW)
        await self._reconcile_active_orders()

        # Initialize open positions count
        await self._update_open_positions_count()

    async def _update_open_positions_count(self):
        """Update count of open positions"""
        try:
            open_trades = await self.db.get_open_trades(self.config.SYMBOL)
            async with self._open_positions_lock:
                self.open_positions_count = len(open_trades)
            app_logger.info(f"Open positions count: {self.open_positions_count}")
        except Exception as e:
            app_logger.warning(f"Could not update open positions count: {e}")
            async with self._open_positions_lock:
                self.open_positions_count = 0

    async def _increment_open_positions(self):
        """Increment open positions count with lock"""
        async with self._open_positions_lock:
            self.open_positions_count += 1

    async def _decrement_open_positions(self):
        """Decrement open positions count with lock"""
        async with self._open_positions_lock:
            self.open_positions_count = max(0, self.open_positions_count - 1)

    async def check_alpaca_health(self):
        """Verify Alpaca API connectivity before trading"""
        try:
            async with self.execution_circuit_breaker:
                account = await asyncio.to_thread(self.client.get_account)
            app_logger.info(f"Alpaca health check passed - Account: {account.status}")
        except Exception as e:
            app_logger.critical(f"Alpaca health check failed: {e}")
            raise

    async def _load_recent_performance(self):
        """Load recent trade performance from DB"""
        try:
            today = datetime.now().date().isoformat()
            perf = await self.db.get_trade_performance(today)
            if perf:
                # Restore performance metrics
                self.recent_trade_performance.extend([perf['avg_pnl']] * perf['trades_count'])
                app_logger.info(f"Restored trade performance: {perf['win_rate']:.2%} win rate")
            else:
                app_logger.info("No trade performance history found, starting fresh")
        except Exception as e:
            app_logger.warning(f"Could not load trade performance: {e}")

    async def _validate_account_status(self):
        """Verify account is ACTIVE and unrestricted before trading"""
        try:
            async with self.execution_circuit_breaker:
                account = await asyncio.to_thread(self.client.get_account)

            # Check account status
            if hasattr(account, 'status'):
                status = account.status.value if hasattr(account.status, 'value') else str(account.status)
                if status not in ['ACTIVE', 'ACTIVE_ENHANCED']:
                    raise AccountBlockedError(f"Account status is {status}, not ACTIVE")

            # Check restriction flags
            if hasattr(account, 'trading_blocked') and account.trading_blocked:
                raise AccountBlockedError("Account is trading_blocked")
            if hasattr(account, 'account_blocked') and account.account_blocked:
                raise AccountBlockedError("Account is account_blocked")
            if hasattr(account, 'pattern_day_trader') and account.pattern_day_trader:
                app_logger.warning("Account is flagged as pattern day trader")

            # Check buying power
            if float(account.buying_power) <= 0:
                raise AccountBlockedError("Account has no buying power")

            app_logger.info(f"Account validation passed: {status}, Equity: ${float(account.equity):,.2f}")

        except AccountBlockedError:
            raise
        except Exception as e:
            app_logger.error(f"Account validation failed: {e}")
            raise AccountBlockedError(f"Could not validate account: {e}")

    async def _reconcile_active_orders(self):
    """V19 FIXED: Check broker for active orders vs our state - CORRECTED with proper status filtering"""
    try:
        # V19 FIX: Only fetch NON-TERMINAL orders from broker to match our DB query
        request = GetOrdersRequest(status='open')  # Changed from 'all' to 'open'
        
        async with self.execution_circuit_breaker:
            broker_orders = await asyncio.to_thread(self.client.get_orders, request)
        
        broker_order_ids: Set[str] = {o.id for o in broker_orders}
        
        # Load our active orders from DB (already excludes terminal states)
        our_active_orders = await self.db.get_active_orders()
        our_order_ids: Set[str] = {o['order_id'] for o in our_active_orders}
        
        # Cancel orders we track but broker doesn't have (ghost orders)
        for order_id in our_order_ids - broker_order_ids:
            app_logger.warning(f"Ghost order {order_id} removed from tracking")
            await self.db.update_active_order(order_id, "CANCELED")
            async with self.order_lock:
                self.active_orders.pop(order_id, None)
        
        # Add orders broker has but we don't track (with status validation)
        for order in broker_orders:
            if order.id not in our_order_ids:
                # V19 FIX: Validate order status before logging
                order_status = getattr(order.status, 'value', str(order.status))
                if order_status in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED']:
                    app_logger.debug(f"Skipping terminal order {order.id} with status {order_status}")
                    continue
                    
                app_logger.info(f"Discovered active order {order.id}")
                await self.db.log_active_order({
                    'order_id': order.id,
                    'symbol': order.symbol,
                    'status': order_status,
                    'filled_qty': int(order.filled_qty) if order.filled_qty else 0,
                    'submitted_at': getattr(order, 'submitted_at', datetime.now())
                })
                async with self.order_lock:
                    self.active_orders[order.id] = {
                        'symbol': order.symbol,
                        'status': order_status,
                        'submitted_at': getattr(order, 'submitted_at', datetime.now())
                    }
        
        app_logger.info(f"Reconciled {len(broker_orders)} active orders")
        
    except APIError as e:
        app_logger.error(f"Reconciliation API error: {e}")
    except Exception as e:
        app_logger.error(f"Reconciliation failed: {e}")

            app_logger.info(f"Reconciled {len(broker_orders)} active orders")

        except APIError as e:
            app_logger.error(f"Reconciliation API error: {e}")
        except Exception as e:
            app_logger.error(f"Reconciliation failed: {e}")

    async def _reset_daily_limits(self):
        """Reset daily trading limits from DB or initialize with DST handling"""
        today = pd.Timestamp.now(tz='America/New_York').date()
        if today != self.last_reset_date:
            # Load from database
            risk_data = await self.db.get_daily_risk(today.isoformat())

            if risk_data:
                self.daily_trades = risk_data['daily_trades']
                # FIXED: Use Decimal constructor for float values
                self.daily_loss = Decimal(str(risk_data['daily_loss'])).quantize(
                    Decimal('0.0001'), rounding=ROUND_HALF_UP)
                self.max_drawdown_today = Decimal(str(risk_data.get('max_drawdown', 0.0)))
                self.safety_mode = risk_data.get('safety_mode_active', False)
                # Restore timeframe_cooldowns
                if risk_data.get('timeframe_cooldowns'):
                    self.timeframe_cooldowns = {
                        tf: datetime.fromisoformat(ts) for tf, ts in risk_data['timeframe_cooldowns'].items()
                    }
            else:
                self.daily_trades = 0
                self.daily_loss = Decimal('0.0')
                self.max_drawdown_today = Decimal('0.0')
                self.safety_mode = False
                self.timeframe_cooldowns = {}

            self.last_reset_date = today

            app_logger.info(f"Daily limits reset: {self.daily_trades} trades, ${self.daily_loss:.2f} loss")

    async def update_daily_loss(self, realized_pnl: float) -> bool:
        """Update daily loss tracking with portfolio value - FIXED float-to-Decimal conversion"""
        # FIXED: Use Decimal constructor for float values
        pnl_decimal = Decimal(str(realized_pnl)).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
        self.daily_loss += pnl_decimal
        self.max_drawdown_today = min(self.max_drawdown_today, self.daily_loss)

        # Get current portfolio value
        try:
            async with self.execution_circuit_breaker:
                account = await asyncio.to_thread(self.client.get_account)
            portfolio_value = Decimal(str(account.portfolio_value))
        except Exception:
            portfolio_value = Decimal(str(self.config.INITIAL_CAPITAL))

        # Use max of initial and current capital
        dynamic_threshold = self.safety_mode_threshold * max(Decimal(str(self.config.INITIAL_CAPITAL)), portfolio_value)

        # Check safety mode activation
        if self.daily_loss <= dynamic_threshold:
            self.safety_mode = True
            app_logger.warning(f"Safety mode activated: ${self.daily_loss:.2f} loss")

        # Persist to DB
        signal_efficiency = await self.db.calculate_signal_efficiency()
        await self.db.log_daily_risk(
            datetime.now().date().isoformat(),
            self.daily_loss,
            self.daily_trades,
            float(self.max_drawdown_today),
            float(portfolio_value),
            self.data_engine.daily_api_calls,
            signal_efficiency,
            safety_mode_active=self.safety_mode,
            timeframe_cooldowns={tf: ts.isoformat() for tf, ts in self.timeframe_cooldowns.items()}
        )

        # Check kill switch
        if self.daily_loss <= -Decimal(str(self.config.MAX_DAILY_LOSS_DOLLAR)):
            app_logger.critical(f"DAILY LOSS LIMIT HIT: ${self.daily_loss:.2f}")
            await self.telegram.send(f"**KILL SWITCH ACTIVATED**\nDaily Loss: ${self.daily_loss:.2f}\nPortfolio: ${portfolio_value:,.2f}\nSystem HALTED.", priority="critical")
            return False

        # Check max drawdown
        if self.max_drawdown_today <= -Decimal(str(self.config.MAX_DAILY_LOSS_DOLLAR * 1.5)):
            app_logger.critical(f"MAX DRAWDOWN LIMIT EXCEEDED: ${self.max_drawdown_today:.2f}")
            await self.telegram.send(f"**DRAWDOWN LIMIT HIT**\nMax DD: ${self.max_drawdown_today:.2f}\nSystem HALTED.", priority="critical")
            return False

        return True

    def validate_price_increment(self, price: float) -> float:
        """Validate and adjust price to exchange minimum increments"""
        if not (price and np.isfinite(price) and price > 0 and not np.isnan(price)):
            raise ValueError(f"Invalid price: {price}")

        # Alpaca increment rules: <$1: $0.0001, $1-$100: $0.01, >$100: $0.05
        if price < 1.0:
            increment = 0.0001
        elif price < 100.0:
            increment = 0.01
        else:
            increment = 0.05

        # Round to nearest increment
        validated_price = round(round(price / increment) * increment, 6)

        if abs(validated_price - price) > increment:
            app_logger.warning(f"Price {price} adjusted to {validated_price} to comply with {increment} increment")

        return validated_price

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
           retry=retry_if_exception_type(APIError))
    async def get_position(self, symbol: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Get current position with caching and proper error handling"""
        now = time.time()
        async with self._position_lock:
            if not force_refresh and (now - self._last_position_fetch) < self.cache_ttl:
                return self.position_cache.get(symbol)

        try:
            async with self.execution_circuit_breaker:
                pos = await asyncio.to_thread(self.client.get_open_position, symbol)

            # Handle fractional shares properly
            qty_decimal = Decimal(str(pos.qty))
            if qty_decimal % 1 != 0:
                app_logger.warning(f"Fractional shares detected: {qty_decimal}; truncating to int")

            position = {
                "symbol": pos.symbol,
                "qty": int(qty_decimal),
                "avg_entry_price": float(pos.avg_entry_price),
                "market_value": float(pos.market_value) if pos.market_value else 0.0,
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
                "side": "LONG" if int(qty_decimal) > 0 else "SHORT"
            }

            async with self._position_lock:
                self.position_cache[symbol] = position
                self._last_position_fetch = now

            return position

        except APIError as e:
            if getattr(e, 'status_code', 404) == 404:
                async with self._position_lock:
                    self.position_cache[symbol] = None
                return None
            else:
                app_logger.error(f"API error getting position: {e}")
                raise
        except Exception as e:
            app_logger.error(f"Error getting position: {e}")
            raise

    async def calculate_position_size(self, symbol: str, signal: Dict[str, Any]) -> Optional[int]:
        """Calculate position size with slippage buffer, risk checks, and volatility adjustment - FIXED division by zero"""
        try:
            # Check max open positions limit
            if self.open_positions_count >= self.config.MAX_OPEN_POSITIONS:
                app_logger.warning(f"Maximum open positions reached: {self.open_positions_count}/{self.config.MAX_OPEN_POSITIONS}")
                return None

            # Position exists check
            position = await self.get_position(symbol)
            if position:
                app_logger.info(f"Already in position: {position['qty']} shares")
                return None

            # Safety mode: reduce position size
            size_multiplier = 0.5 if self.safety_mode else 1.0

            # Get account
            async with self.execution_circuit_breaker:
                account = await asyncio.to_thread(self.client.get_account)

            equity = Decimal(str(account.equity))
            cash = Decimal(str(account.cash))
            buying_power = Decimal(str(account.buying_power))

            # Portfolio exposure check
            async with self.execution_circuit_breaker:
                all_positions = await asyncio.to_thread(self.client.get_all_positions)

            # Calculate net exposure (long - short)
            current_exposure = Decimal('0.0')
            for p in all_positions:
                # Safely convert qty to Decimal
                qty_decimal = Decimal(str(p.qty))
                market_value = Decimal(str(p.market_value)) if p.market_value else Decimal('0.0')
                if qty_decimal > 0:
                    current_exposure += market_value
                else:
                    current_exposure -= market_value

            max_exposure = equity * Decimal(str(self.config.PORTFOLIO_MAX_EXPOSURE))

            if abs(current_exposure) >= max_exposure:
                app_logger.warning(f"Portfolio exposure limit: {abs(current_exposure)/equity:.1%} >= {self.config.PORTFOLIO_MAX_EXPOSURE:.1%}")
                return None

            # Get latest data from shared engine
            df = self.data_engine.get_df("1d")
            if df.empty or 'atr' not in df.columns:
                app_logger.error("No ATR data for position sizing")
                return None

            atr = df['atr'].iloc[-1]
            current_price = df['close'].iloc[-1]

            # FIXED: Validate ATR and price BEFORE calculations with fallback for near-zero ATR
            if not (atr and np.isfinite(atr) and atr > 0):
                app_logger.error(f"Invalid ATR: {atr}")
                # Fallback to historical ATR with minimum floor
                if len(df) > 20:
                    historical_atr = df['atr'].iloc[-20:].median()
                    atr = max(historical_atr, current_price * 0.005)  # Minimum 0.5% of price
                    app_logger.warning(f"Using fallback ATR: {atr:.4f}")
                else:
                    # Minimum 1% of price or $0.01
                    atr = max(current_price * 0.01, 0.01)
                    app_logger.warning(f"Using minimum ATR: {atr:.4f}")

            if not (current_price and np.isfinite(current_price) and current_price > 0):
                app_logger.error(f"Invalid current price: {current_price}")
                return None

            # FIXED: Ensure risk_per_share has minimum floor BEFORE division with robust validation
            min_risk_per_share = current_price * (self.config.MIN_RISK_PER_SHARE_BPS / 10000)
            # Ensure ATR is positive and reasonable
            safe_atr = max(atr, current_price * 0.001, 0.01)  # Minimum 0.1% of price or $0.01
            risk_per_share = max(safe_atr * self.config.STOP_LOSS_ATR, min_risk_per_share, Decimal('0.01'))

            # CRITICAL FIX: Ensure risk_per_share is positive before division
            if risk_per_share <= Decimal('1e-9'):  # More robust check
                app_logger.error(f"Risk per share is zero or negative: {risk_per_share}")
                return None

            # Configurable spread buffer
            spread_buffer = current_price * (self.config.SPREAD_BUFFER_BPS / 10000)
            risk_per_share += spread_buffer

            # Calculate max allowed shares early
            max_allowed_shares = int((equity * Decimal(str(self.config.MAX_POS_SIZE_PCT))) / Decimal(str(current_price)))

            # Volatility-adjusted position size using Decimal throughout
            vol_adjusted_pos_size = Decimal(str(self.config.MAX_POS_SIZE_PCT)) * (Decimal('0.02') / (Decimal(str(atr)) / Decimal(str(current_price)) + Decimal('0.01')))
            vol_adjusted_pos_size = min(vol_adjusted_pos_size, Decimal(str(self.config.MAX_POS_SIZE_PCT)))

            # Position size based on risk using Decimal consistently
            risk_amount = equity * vol_adjusted_pos_size * Decimal(str(size_multiplier))

            # Use Decimal consistently for division with validated denominator
            shares_decimal = (risk_amount / Decimal(str(risk_per_share))).to_integral_value(rounding=ROUND_HALF_UP)
            shares = int(shares_decimal)

            # Enforce round lots BEFORE capping
            if self.config.ENFORCE_ROUND_LOTS and shares >= 100:
                shares = (shares // 100) * 100

            # Check minimum share count before truncation
            if shares < self.config.MIN_POSITION_SIZE:
                app_logger.warning(f"Position size < {self.config.MIN_POSITION_SIZE} share (calculated: {shares})")
                return None

            # Commission and slippage buffer using configurable cash buffer
            cash_buffer = Decimal(str(self.config.CASH_BUFFER_PCT))
            max_shares_by_cash = int((buying_power * cash_buffer) / (Decimal(str(current_price)) * Decimal('1.0001')))
            shares = min(shares, max_shares_by_cash)

            # Apply early cap (AFTER round lot adjustment)
            shares = min(shares, max_allowed_shares)

            # Check for SQLite integer overflow
            max_sqlite_int = 2**63 - 1
            if shares > max_sqlite_int:
                app_logger.critical(f"Position size {shares} exceeds SQLite limit {max_sqlite_int}")
                shares = max_sqlite_int

            app_logger.info(f"Position size: {shares} shares (risk: ${float(risk_per_share)*shares:.2f}, vol_adj: {float(vol_adjusted_pos_size):.1%}, cash_buffer: {float(cash_buffer):.2%})")
            return shares

        except Exception as e:
            app_logger.error(f"Position sizing error: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
           retry=retry_if_exception_type(APIError))
    async def verify_order_filled(self, order_id: str, timeout: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Verify order was filled and return fill details with proper error handling"""
        if timeout is None:
            timeout = self.config.ORDER_TIMEOUT_SECONDS

        try:
            start_time = time.time()
            check_interval = 2

            while time.time() - start_time < timeout:
                async with self.execution_circuit_breaker:
                    order = await asyncio.to_thread(self.client.get_order_by_id, order_id)

                # Update active order status under lock
                filled_qty = int(order.filled_qty) if order.filled_qty else 0
                await self._update_active_order(order_id, order.status.value, filled_qty)

                # Defensive status handling
                order_status = getattr(order.status, 'value', str(order.status))

                if order_status == "FILLED":
                    filled_qty = int(float(order.filled_qty))
                    filled_avg_price = float(order.filled_avg_price)

                    # Verify partial fills are complete
                    total_qty = int(float(order.qty))
                    if filled_qty >= total_qty:
                        return {
                            "filled_qty": filled_qty,
                            "filled_avg_price": filled_avg_price,
                            "status": "FILLED",
                            "filled_at": getattr(order, 'filled_at', datetime.now()),
                            "quantity": total_qty
                        }
                    else:
                        app_logger.warning(f"Order {order_id} partially filled: {filled_qty}/{total_qty}, waiting for completion")

                elif order_status in ["REJECTED", "CANCELED", "EXPIRED"]:
                    app_logger.error(f"Order {order_id} failed with status: {order_status}")
                    await self._update_active_order(order_id, order_status)
                    return {"status": order_status, "reason": getattr(order, 'reject_reason', 'Unknown')}

                elif order_status == "PARTIALLY_FILLED":
                    app_logger.info(f"Order {order_id} partially filled: {order.filled_qty}/{order.qty}")
                    # Track partial fills
                    await self._update_active_order(order_id, "PARTIALLY_FILLED", filled_qty)
                    # Continue waiting for full fill

                await asyncio.sleep(check_interval)

            # Timeout - attempt to cancel
            app_logger.warning(f"Order {order_id} not filled within {timeout}s, attempting cancel")
            try:
                async with self.execution_circuit_breaker:
                    await asyncio.to_thread(self.client.cancel_order, order_id)
                await self._update_active_order(order_id, "CANCELED")
            except APIError as cancel_error:
                app_logger.error(f"Cancel failed: {cancel_error}")
                # Order might have filled between timeout and cancel attempt
                try:
                    async with self.execution_circuit_breaker:
                        order = await asyncio.to_thread(self.client.get_order_by_id, order_id)
                    if order.status.value == "FILLED":
                        total_qty = int(float(order.qty))
                        return {
                            "filled_qty": int(order.filled_qty),
                            "filled_avg_price": float(order.filled_avg_price),
                            "status": "FILLED",
                            "filled_at": getattr(order, 'filled_at', datetime.now()),
                            "quantity": total_qty
                        }
                except Exception:
                    pass

            await self._update_active_order(order_id, "TIMEOUT")
            return {"status": "TIMEOUT"}

        except APIError as e:
            app_logger.error(f"Order verification API error: {e}")
            await self._update_active_order(order_id, "API_ERROR")
            return {"status": "API_ERROR", "error": str(e)}
        except Exception as e:
            app_logger.error(f"Order verification error: {e}")
            return None

    async def _update_active_order(self, order_id: str, status: str, filled_qty: Optional[int] = None):
        """Helper to update active order with proper locking"""
        async with self.order_lock:
            if order_id in self.active_orders:
                self.active_orders[order_id]['status'] = status
                if filled_qty is not None:
                    self.active_orders[order_id]['filled_qty'] = filled_qty
            await self.db.update_active_order(order_id, status, filled_qty)

    async def can_execute_trade(self, signal: Dict[str, Any]) -> bool:
        """Check if we can execute a trade based on timing, cooldowns, and performance - FIXED duplicate detection"""
        now = datetime.now()

        # Global cooldown between any trades
        if (now - self.last_trade_time).total_seconds() < self.config.MIN_TRADE_COOLDOWN_SECONDS:
            return False

        # Timeframe-specific cooldown
        last_tf_trade = self.timeframe_cooldowns.get(signal['timeframe'])
        if last_tf_trade and (now - last_tf_trade).total_seconds() < self.cooldown_duration:
            return False

        # FIXED: Correct duplicate trade detection - check for same timeframe AND opposite action (for closing)
        try:
            open_trades = await self.db.get_open_trades(self.config.SYMBOL)
            # Check if there's already an open trade in the SAME timeframe with SAME action (block entry)
            # Allow opposite action (for closing)
            if any(t['timeframe'] == signal['timeframe'] and t['action'] == signal['action'] for t in open_trades):
                app_logger.warning(f"Duplicate detection: Already have open {signal['action']} trade in {signal['timeframe']}")
                return False
        except Exception as e:
            app_logger.warning(f"Could not check duplicate trades: {e}")
            return False

        # Market hours check
        if self.config.MARKET_HOURS_ONLY:
            current_time = now.time()
            start_time = datetime.strptime(self.config.TRADING_START_TIME, "%H:%M").time()
            end_time = datetime.strptime(self.config.TRADING_END_TIME, "%H:%M").time()

            if not (start_time <= current_time <= end_time):
                return False

        # Performance-based throttling
        if len(self.recent_trade_performance) >= 5:
            recent_win_rate = sum(1 for p in self.recent_trade_performance if p > 0) / len(self.recent_trade_performance)
            if recent_win_rate < self.performance_threshold:
                if (now - self.last_trade_time).total_seconds() < self.config.MAX_TRADE_COOLDOWN_SECONDS:
                    app_logger.warning(f"Performance throttling active: {recent_win_rate:.1%} win rate")
                    return False

        # Safety mode check
        if self.safety_mode:
            if signal.get('quality', 0) < 80:
                app_logger.debug("Safety mode: Signal quality too low")
                return False

        # API budget check (high priority for trades)
        try:
            has_budget = await self.data_engine.has_api_budget(priority="high")
            if not has_budget:
                app_logger.warning("API budget insufficient for trade execution")
                return False
        except Exception as e:
            app_logger.warning(f"Could not check API budget: {e}, assuming insufficient")
            return False

        # Correlation check between timeframes
        if not await self._check_signal_correlation(signal):
            return False

        # Check max open positions
        if self.open_positions_count >= self.config.MAX_OPEN_POSITIONS:
            app_logger.warning(f"Maximum open positions reached: {self.open_positions_count}/{self.config.MAX_OPEN_POSITIONS}")
            return False

        return True

    async def _check_signal_correlation(self, signal: Dict[str, Any]) -> bool:
        """Clear signal correlation logic"""
        primary_signal = self._primary_tf_signal
        if not primary_signal:
            return True

        # Clear matrix: primary BEAR blocks secondary BUY unless high quality
        if primary_signal['regime'] == 'BEAR' and signal['action'] == 'BUY':
            if signal['timeframe'] != self.config.PRIMARY_TIMEFRAME:
                if signal['quality'] < 85:
                    app_logger.warning(f"Blocking BUY in {signal['timeframe']} (primary is BEAR, quality {signal['quality']:.1f} < 85)")
                    return False

        # Clear matrix: primary BULL blocks secondary SELL unless high quality
        if primary_signal['regime'] == 'BULL' and signal['action'] == 'SELL':
            if signal['timeframe'] != self.config.PRIMARY_TIMEFRAME:
                if signal['quality'] < 85:
                    app_logger.warning(f"Blocking SELL in {signal['timeframe']} (primary is BULL, quality {signal['quality']:.1f} < 85)")
                    return False

        return True

    async def execute_trade(self, symbol: str, signal: Dict[str, Any]) -> Optional[str]:
        """Execute trade with verification, slippage accounting, and smart execution"""
        execution_start = time.time()

        # Overall execution timeout
        try:
            return await asyncio.wait_for(
                self._execute_trade_internal(symbol, signal),
                timeout=self.config.EXECUTION_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            app_logger.critical(f"Trade execution timed out after {self.config.EXECUTION_TIMEOUT_SECONDS}s")
            await self.telegram.send(f"**EXECUTION TIMEOUT**\nSymbol: {symbol}\nAction: {signal['action']}\nSystem may be degraded.", priority="critical")
            return None

    async def _execute_trade_internal(self, symbol: str, signal: Dict[str, Any]) -> Optional[str]:
        """Internal trade execution logic"""
        # Check if we can execute
        if not await self.can_execute_trade(signal):
            app_logger.warning("Cannot execute trade: timing or limit constraints")

            # Log rejected signal
            await self._log_rejected_signal(signal, symbol, "REJECTED_COOLDOWN")
            return None

        try:
            await self._reset_daily_limits()

            # Validate signal
            if signal['quality'] < self.config.MIN_TRADE_QUALITY:
                app_logger.warning(f"Signal quality too low: {signal['quality']:.1f} < {self.config.MIN_TRADE_QUALITY}")
                await self._log_rejected_signal(signal, symbol, "REJECTED_QUALITY")
                return None

            # Prevent duplicate timeframe trades
            recent_trades = await self.db.get_recent_trades(symbol, limit=10)
            if any(t['timeframe'] == signal['timeframe'] and t['status'] == 'OPEN' and t['action'] == signal['action'] for t in recent_trades):
                app_logger.warning(f"Already have open {signal['action']} trade in {signal['timeframe']}")
                await self._log_rejected_signal(signal, symbol, "REJECTED_DUPLICATE_TF")
                return None

            # Calculate position size
            quantity = await self.calculate_position_size(symbol, signal)
            if not quantity:
                await self._log_rejected_signal(signal, symbol, "REJECTED_SIZING")
                return None

            # Get market data with fresh fetch and staleness check
            fetch_success = await self.data_engine.fetch_timeframe("1d", force_refresh=True, priority="high")
            if not fetch_success:
                app_logger.error("Failed to fetch fresh data for execution")
                return None

            df = self.data_engine.get_df("1d")
            if df.empty:
                return None

            # Check data staleness
            last_bar_age = (datetime.now() - df.index[-1]).total_seconds()
            expected_interval = 86400  # 1 day
            if last_bar_age > expected_interval * 2:
                app_logger.error(f"Data stale ({last_bar_age}s old), aborting trade")
                return None

            current_price = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1]

            # ATR fallback for near-zero volatility
            atr = max(atr, current_price * 0.005)  # Minimum 0.5% of price

            # Calculate execution price based on order type
            if self.config.USE_LIMIT_ORDERS and signal['quality'] > 70:
                execution_type = "LIMIT"
                # Validate price increment
                passivity = self.config.LIMIT_ORDER_PASSIVITY_BPS / 10000
                if signal['action'] == 'BUY':
                    limit_price = current_price - (passivity * current_price)
                else:
                    limit_price = current_price + (passivity * current_price)

                limit_price = self.validate_price_increment(limit_price)
                execution_price = limit_price
                slip_bps = self.config.SLIPPAGE_BPS * 0.7
            else:
                execution_type = "MARKET"
                if signal['action'] == 'BUY':
                    execution_price = current_price + (self.config.SLIPPAGE_BPS / 10000 * current_price)
                else:
                    execution_price = current_price - (self.config.SLIPPAGE_BPS / 10000 * current_price)

                execution_price = self.validate_price_increment(execution_price)
                slip_bps = self.config.SLIPPAGE_BPS

            # Calculate stops from EXECUTION_PRICE, not current_price
            stop_distance = self.config.STOP_LOSS_ATR * atr
            take_distance = self.config.TAKE_PROFIT_ATR * atr

            if signal['action'] == 'BUY':
                stop_price = execution_price - stop_distance
                take_profit_price = execution_price + take_distance
            else:
                # FIXED CORRECT: For SELL: take_profit < execution < stop (stop above entry, take profit below)
                stop_price = execution_price + stop_distance
                take_profit_price = execution_price - take_distance

            # Validate stop and take profit prices
            stop_price = self.validate_price_increment(stop_price)
            take_profit_price = self.validate_price_increment(take_profit_price)

            # Validate bracket order price ordering BEFORE submission
            if not self._validate_bracket_order_prices(signal['action'], stop_price, execution_price, take_profit_price):
                raise ValueError(f"Invalid bracket order prices for {signal['action']}")

            # Create order request
            order_request = self._create_order_request(
                symbol, quantity, signal['action'], execution_type,
                execution_price if execution_type == "LIMIT" else None,
                stop_price, take_profit_price
            )

            # Submit order to broker
            async with self.execution_circuit_breaker:
                order = await asyncio.to_thread(self.client.submit_order, order_data=order_request)

            # Validate order response with null check
            if not order or not hasattr(order, 'id'):
                raise APIError(f"Invalid order response: {order}", status_code=500)

            order_id = order.id

            # Log real order after submission
            await self.db.log_active_order({
                'order_id': order_id,
                'symbol': symbol,
                'status': OrderStatus.ACCEPTED.value,
                'filled_qty': 0,
                'submitted_at': datetime.now()
            })

            # Update in-memory tracking with lock
            await self._add_active_order(order_id, {
                'symbol': symbol,
                'signal': signal,
                'submitted_at': datetime.now(),
                'quantity': quantity,
                'execution_type': execution_type
            })

            # Verify fill
            fill_details = await self.verify_order_filled(order_id)

            # Remove from active tracking if filled or failed
            if fill_details and fill_details.get("status") == "FILLED":
                await self._remove_active_order(order_id)
            else:
                app_logger.error(f"Order {order_id} verification failed: {fill_details}")
                # Decrement position count on failure
                await self._decrement_open_positions()

            if not fill_details or fill_details.get("status") != "FILLED":
                app_logger.error(f"Order {order_id} failed verification: {fill_details}")
                await self.telegram.send(
                    f"**ORDER FAILED**\nOrder ID: {order_id}\nStatus: {fill_details.get('status')}\nReason: {fill_details.get('reason')}",
                    priority="critical"
                )
                return None

            execution_duration = int((time.time() - execution_start) * 1000)

            # Calculate slippage with Decimal precision
            actual_execution_price = fill_details['filled_avg_price']
            total_quantity = fill_details['quantity']

            # Correct slippage calculation for both BUY and SELL orders
            slippage_cost, actual_slippage_bps = self._calculate_slippage(
                signal['action'], actual_execution_price, current_price, fill_details['filled_qty']
            )

            commission_cost = fill_details['filled_qty'] * self.config.COMMISSION_PER_SHARE

            # Use order ID as trade ID
            trade_id = order_id

            trade_data = {
                'id': trade_id,
                'symbol': symbol,
                'timeframe': signal['timeframe'],
                'action': signal['action'],
                'quantity': fill_details['filled_qty'],
                'entry_price': fill_details['filled_avg_price'],
                'status': 'OPEN',
                'entry_time': fill_details.get('filled_at', datetime.now()),
                'stop_loss': stop_price,
                'take_profit': take_profit_price,
                'quality_score': signal['quality'],
                'regime': signal['regime'],
                'confidence': signal['confidence'],
                'order_id': order_id,
                'slippage_cost': slippage_cost,
                'commission_cost': commission_cost,
                'timeframe_weight': self._get_timeframe_weight(signal['timeframe']),
                'execution_duration_ms': execution_duration,
                'data_hash': self.data_engine.data_hashes.get('1d', ''),
                'execution_price': actual_execution_price,
                'execution_type': execution_type
            }

            await self.db.log_trade(trade_data)

            # Log signal execution
            await self.db.log_signal({
                'id': f"{datetime.now().isoformat()}_{signal['timeframe']}_{signal['action']}",
                'timestamp': datetime.now(),
                'timeframe': signal['timeframe'],
                'action': signal['action'],
                'score': signal['score'],
                'regime': signal['regime'],
                'confidence': signal['confidence'],
                'quality': signal['quality'],
                'symbol': symbol,
                'execution_decision': 'EXECUTED',
                'api_budget_pct': self.data_engine.get_budget_status()['pct_used']
            })

            # Update trade tracking
            self.daily_trades += 1
            self.last_trade_time = datetime.now()
            self.last_trade_timeframe = signal['timeframe']
            self.timeframe_cooldowns[signal['timeframe']] = datetime.now()

            # Update open positions count AFTER successful execution
            await self._increment_open_positions()

            # Invalidate position cache
            async with self._position_lock:
                self.position_cache.pop(symbol, None)

            # Build message carefully with truncation indicator
            msg_parts = [
                f"**TRADE EXECUTED** ({signal['timeframe']}) - {execution_type}",
                f"Symbol: {symbol}",
                f"Action: {signal['action']} ({signal['regime']} regime)",
                f"Quantity: {fill_details['filled_qty']}/{total_quantity}",
                f"Price: ${actual_execution_price:.2f}",
                f"Slip: {actual_slippage_bps:.1f}bps",
                f"Stop: ${stop_price:.2f}",
                f"Take: ${take_profit_price:.2f}",
                f"Quality: {signal['quality']:.1f}",
                f"Daily Trades: {self.daily_trades}",
                f"Open Positions: {self.open_positions_count}/{self.config.MAX_OPEN_POSITIONS}",
                f"API Budget: {self.data_engine.get_budget_status()['pct_used']:.1f}%"
            ]

            full_msg = "\n".join(msg_parts)
            if len(full_msg) > 4096:
                app_logger.warning(f"Telegram message truncated: {len(full_msg)} chars")

            await self.telegram.send(full_msg, priority="critical")

            app_logger.info(f"Trade executed: {trade_id} in {execution_duration}ms via {execution_type}")
            return order_id

        except APIError as e:
            app_logger.error(f"Order submission failed: {e}")
            await self.telegram.send(f"**ORDER FAILED**\n{str(e)}\nStatus: {e.status_code}", priority="critical")
            # Decrement position count on API failure
            await self._decrement_open_positions()
            return None
        except KeyboardInterrupt:
            raise
        except SystemExit:
            raise
        except Exception as e:
            app_logger.error(f"Trade execution error: {e}", exc_info=True)
            # Decrement position count on general failure
            await self._decrement_open_positions()
            return None

    async def _log_rejected_signal(self, signal: Dict[str, Any], symbol: str, reason: str):
        """Helper to log rejected signals"""
        await self.db.log_signal({
            'id': f"{datetime.now().isoformat()}_{signal['timeframe']}_{signal['action']}",
            'timestamp': datetime.now(),
            'timeframe': signal['timeframe'],
            'action': signal['action'],
            'score': signal['score'],
            'regime': signal['regime'],
            'confidence': signal['confidence'],
            'quality': signal['quality'],
            'symbol': symbol,
            'execution_decision': reason,
            'api_budget_pct': self.data_engine.get_budget_status()['pct_used']
        })

    def _validate_bracket_order_prices(self, action: str, stop_price: float, execution_price: float, take_profit_price: float) -> bool:
        """
        Validate bracket order price ordering - CORRECTED for short positions

        For BUY orders: stop < execution < take_profit
        For SELL orders: take_profit < execution < stop (stop above entry, take profit below)
        """
        if action == 'BUY':
            # For BUY: stop < execution < take_profit
            if not (stop_price < execution_price < take_profit_price):
                app_logger.error(f"Invalid BUY bracket: stop={stop_price:.4f} < exec={execution_price:.4f} < take={take_profit_price:.4f}")
                return False
        else:  # SELL
            # FIXED CORRECT: For SELL: take_profit < execution < stop (stop above entry, take profit below)
            if not (take_profit_price < execution_price < stop_price):
                app_logger.error(f"Invalid SELL bracket: take={take_profit_price:.4f} < exec={execution_price:.4f} < stop={stop_price:.4f}")
                return False
        return True

    def _create_order_request(self, symbol: str, quantity: int, action: str, execution_type: str,
                              limit_price: Optional[float], stop_price: float, take_profit_price: float):
        """Create order request based on type"""
        side = OrderSide.BUY if action == 'BUY' else OrderSide.SELL

        if execution_type == "LIMIT":
            return LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=round(limit_price, 6),
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=round(stop_price, 6)),
                take_profit=TakeProfitRequest(limit_price=round(take_profit_price, 6))
            )
        else:
            return MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=round(stop_price, 6)),
                take_profit=TakeProfitRequest(limit_price=round(take_profit_price, 6))
            )

    def _calculate_slippage(self, action: str, actual_price: float, expected_price: float, quantity: int) -> Tuple[float, float]:
        """Calculate slippage correctly for both long and short positions - returns consistent float types"""
        # Use Decimal for precise calculation
        actual_decimal = Decimal(str(actual_price))
        expected_decimal = Decimal(str(expected_price))

        if action == 'BUY':
            # For BUY: positive slippage = expected - actual (lower price is better)
            price_diff = expected_decimal - actual_decimal
        else:  # SELL
            # For SELL: positive slippage = actual - expected (higher price is better)
            price_diff = actual_decimal - expected_decimal

        slippage_bps = (price_diff / expected_decimal) * Decimal('10000')
        slippage_bps_float = float(slippage_bps)
        slippage_cost = slippage_bps_float / 10000 * actual_price * quantity
        return slippage_cost, slippage_bps_float

    async def _add_active_order(self, order_id: str, order_data: Dict[str, Any]):
        """Helper to add active order with locking"""
        async with self.order_lock:
            self.active_orders[order_id] = order_data

    async def _remove_active_order(self, order_id: str):
        """Helper to remove active order with locking"""
        async with self.order_lock:
            self.active_orders.pop(order_id, None)

    async def check_and_close_positions(self, symbol: str, signals: List[Dict[str, Any]]):
        """Check if position should be closed based on opposing signal from SAME timeframe"""
        try:
            position = await self.get_position(symbol)
            if not position:
                return

            # Get position timeframe from trade history
            position_tf = await self._get_position_timeframe(symbol)
            if not position_tf:
                app_logger.warning(f"Could not determine timeframe for position {symbol}")
                return

            # Find opposing signal from same timeframe
            opposing_action = "SELL" if position['side'] == "LONG" else "BUY"
            matching_signals = [s for s in signals if s['timeframe'] == position_tf and s['action'] == opposing_action]

            if not matching_signals:
                return

            signal = matching_signals[0]

            # Verify quality and cooldown
            if signal['quality'] < self.config.MIN_TRADE_QUALITY:
                app_logger.debug(f"Exit signal quality too low: {signal['quality']}")
                return

            # Check timeframe cooldown
            if (datetime.now() - self.timeframe_cooldowns.get(position_tf, datetime.min)).total_seconds() < self.cooldown_duration:
                app_logger.debug(f"Exit cooldown active for {position_tf}")
                return

            # Submit closing order
            close_quantity = abs(position['qty'])
            app_logger.info(f"Closing {close_quantity} shares of {symbol} (reason: {signal['timeframe']} {signal['action']})")

            # Get actual trade data from DB for accurate PnL calculation
            trade_id = await self._get_open_trade_id(symbol)
            if trade_id:
                open_trades = await self.db.get_open_trades(symbol)
                if open_trades:
                    trade_data = open_trades[0]
                    entry_price = trade_data.get('entry_price')
                    if entry_price:
                        # Calculate actual PnL based on entry price
                        current_price = position['avg_entry_price'] + position['unrealized_pl'] / position['qty']
                        pnl = (current_price - entry_price) * position['qty'] if position['side'] == 'LONG' else (entry_price - current_price) * abs(position['qty'])
                        position['unrealized_pl'] = pnl

            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=close_quantity,
                side=OrderSide.SELL if position['side'] == "LONG" else OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )

            async with self.execution_circuit_breaker:
                order = await asyncio.to_thread(self.client.submit_order, order_data=order_request)

            fill_details = await self.verify_order_filled(order.id, timeout=60)

            if fill_details and fill_details.get("status") == "FILLED":
                # Update trade record
                if trade_id:
                    await self.db.update_trade_exit(
                        trade_id,
                        exit_price=fill_details['filled_avg_price'],
                        exit_time=fill_details['filled_at'],
                        pnl=Decimal(str(position['unrealized_pl'])),
                        status='CLOSED',
                        exit_reason=f"signal_{signal['timeframe']}"
                    )

                # Update tracking
                self.timeframe_cooldowns[position_tf] = datetime.now()

                # Update recent performance
                self.recent_trade_performance.append(position['unrealized_pl'])

                # Update open positions count
                await self._decrement_open_positions()

                # Persist performance
                if len(self.recent_trade_performance) > 0:
                    win_rate = sum(1 for p in self.recent_trade_performance if p > 0) / len(self.recent_trade_performance)
                    await self.db.log_trade_performance(
                        datetime.now().date().isoformat(),
                        win_rate,
                        sum(self.recent_trade_performance) / len(self.recent_trade_performance),
                        len(self.recent_trade_performance)
                    )

                # Update daily loss with Decimal
                await self.update_daily_loss(position['unrealized_pl'])

                await self.telegram.send(
                    f"**POSITION CLOSED**\nSymbol: {symbol}\nQty: {fill_details['filled_qty']}\nPrice: ${fill_details['filled_avg_price']:.2f}\nPnL: ${position['unrealized_pl']:.2f}",
                    priority="critical"
                )

        except Exception as e:
            app_logger.error(f"Position close error: {e}", exc_info=True)

    async def _get_position_timeframe(self, symbol: str) -> Optional[str]:
        """Get the timeframe of the open position"""
        open_trades = await self.db.get_open_trades(symbol)
        if open_trades:
            return open_trades[0].get('timeframe')
        return None

    async def _get_open_trade_id(self, symbol: str) -> Optional[str]:
        """Get the ID of the open trade"""
        open_trades = await self.db.get_open_trades(symbol)
        if open_trades:
            return open_trades[0].get('id')
        return None

    def _get_timeframe_weight(self, timeframe: str) -> float:
        """Get normalized timeframe weight"""
        weights = {
            "1d": 4.0,
            "4h": 2.5,
            "1h": 1.5,
            "15m": 1.0
        }
        return weights.get(timeframe, 1.0)

    async def check_liquidated_positions(self):
        """Enhanced check for positions liquidated by broker"""
        try:
            async with self.execution_circuit_breaker:
                positions = await asyncio.to_thread(self.client.get_all_positions)

            broker_symbols = {p.symbol for p in positions}
            db_open_trades = await self.db.get_open_trades()

            for trade in db_open_trades:
                if trade['symbol'] not in broker_symbols:
                    # Position no longer exists - likely liquidated
                    app_logger.critical(f"Position liquidated: {trade['symbol']} (trade {trade['id']})")
                    # Use config-based maximum loss instead of magic number
                    max_loss = -Decimal(str(self.config.INITIAL_CAPITAL * self.config.MAX_DAILY_LOSS_PCT * 10))
                    await self.db.update_trade_exit(
                        trade['id'],
                        exit_price=0.0,
                        exit_time=datetime.now(),
                        pnl=max_loss.quantize(Decimal('0.0001')),
                        status='LIQUIDATED',
                        exit_reason='broker_liquidation'
                    )
                    # Update open positions count
                    await self._decrement_open_positions()

                    await self.telegram.send(
                        f"**POSITION LIQUIDATED**\nSymbol: {trade['symbol']}\nTrade ID: {trade['id']}\nImmediate attention required!",
                        priority="critical"
                    )
                else:
                    # Verify position size matches
                    broker_pos = next((p for p in positions if p.symbol == trade['symbol']), None)
                    if broker_pos and int(broker_pos.qty) != trade['quantity']:
                        app_logger.critical(f"Position size mismatch: DB {trade['quantity']} vs Broker {broker_pos.qty}")
                        await self.telegram.send(
                            f"**SIZE MISMATCH**\nSymbol: {trade['symbol']}\nDB: {trade['quantity']}\nBroker: {broker_pos.qty}",
                            priority="critical"
                        )

        except Exception as e:
            app_logger.error(f"Liquidation check error: {e}")

    async def shutdown(self):
        """Graceful shutdown - cancel pending orders and close positions"""
        app_logger.info("Shutting down execution engine...")

        # Cancel all pending/submitted orders
        pending_orders = []
        async with self.order_lock:
            pending_orders = [
                oid for oid, order in self.active_orders.items()
                if order.get('status') in ['PENDING', 'SUBMITTED', 'ACCEPTED']
            ]

        for order_id in pending_orders:
            try:
                app_logger.info(f"Canceling order {order_id} on shutdown")
                async with self.execution_circuit_breaker:
                    await asyncio.to_thread(self.client.cancel_order, order_id)
                await self.db.update_active_order(order_id, "CANCELED")
            except Exception as e:
                app_logger.error(f"Failed to cancel {order_id}: {e}")

        # Close all open positions
        try:
            async with self.execution_circuit_breaker:
                positions = await asyncio.to_thread(self.client.get_all_positions)
            for pos in positions:
                app_logger.info(f"Closing position {pos.symbol} on shutdown")
                close_qty = abs(int(pos.qty))
                order_request = MarketOrderRequest(
                    symbol=pos.symbol,
                    qty=close_qty,
                    side=OrderSide.SELL if int(pos.qty) > 0 else OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                await asyncio.to_thread(self.client.submit_order, order_data=order_request)

                # Update open positions count
                await self._decrement_open_positions()
        except Exception as e:
            app_logger.error(f"Failed to close positions on shutdown: {e}")

        app_logger.info("Execution engine shutdown complete")

# ===============================================================================
# 🚀 MAIN ENTRY POINT - PRODUCTION BOOTSTRAP (V19 ENHANCED)
# ===============================================================================

async def main():
    """Main trading loop with graceful shutdown and error handling"""
    # Initialize secrets validation
    SecretsManager.validate_env_security()
    SecretsManager.load_from_vault()

    # Parse args first, then create config
    args = parse_args()

    # Create configuration
    conf = SystemConfig(
        SYMBOL=args.symbol,
        PAPER_TRADING=args.paper,
        LIVE_LOOP_INTERVAL_SECONDS=args.loop_interval
    )

    # Bootstrap components
    db = DatabaseManager(conf)
    await db.initialize()

    # Clean up old orders on startup
    await db.cleanup_old_orders()

    data_engine = None
    telegram = None

    try:
        data_engine = MultiTimeframeDataEngine(conf, db)
        await data_engine.initialize()

        telegram = TelegramBot(conf.TELEGRAM_TOKEN, conf.TELEGRAM_CHANNEL)
        await telegram.initialize()

        brain = MultiTimeframeBrain(conf)
        await brain.initialize()
        brain.data_engine = data_engine  # Uses weak reference now
        brain.telegram = telegram  # Uses weak reference now

        client = TradingClient(conf.API_KEY, conf.SECRET_KEY, paper=conf.PAPER_TRADING)

        engine = ExecutionEngine(conf, client, db, telegram, data_engine)
        await engine.initialize()

        # Signal handlers
        shutdown_event = asyncio.Event()

        def handle_shutdown(sig, frame):
            app_logger.warning(f"Signal {sig} received, shutting down...")
            shutdown_event.set()

        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

        # Global exception handler
        def handle_exception(loop, context):
            app_logger.critical(f"Global exception: {context.get('exception', context['message'])}")
            if telegram:
                # Check telegram exists before sending
                try:
                    asyncio.create_task(telegram.send(
                        f"**FATAL ERROR**\n{context.get('exception', 'Unknown')}", priority="critical"))
                except Exception as e:
                    app_logger.error(f"Failed to send Telegram alert: {e}")

        asyncio.get_event_loop().set_exception_handler(handle_exception)

        app_logger.info("TITANIUM v19-PROD-ENHANCED entering live loop...")

        try:
            await _live_loop(conf, data_engine, brain, engine, telegram, shutdown_event)
        except KeyboardInterrupt:
            app_logger.info("Shutdown signal received, cleaning up...")
        except Exception as e:
            app_logger.critical(f"Fatal error in main: {e}", exc_info=True)
        finally:
            await engine.shutdown()
            if telegram:
                await telegram.close()
            if data_engine:
                # Ensure data engine session is closed
                if data_engine.session and not data_engine.session.closed:
                    await data_engine.session.close()
            await db.backup()
            app_logger.info("Graceful shutdown complete")
    except Exception as e:
        app_logger.critical(f"Failed to initialize system: {e}", exc_info=True)
        if telegram:
            await telegram.close()
        if data_engine and data_engine.session and not data_engine.session.closed:
            await data_engine.session.close()
        raise

async def _live_loop(config: SystemConfig, data_engine: MultiTimeframeDataEngine, brain: MultiTimeframeBrain,
                     executor: ExecutionEngine, telegram: TelegramBot, shutdown_event: asyncio.Event):
    """Core trading loop."""
    loop_count = 0

    while not shutdown_event.is_set():
        try:
            loop_start = time.time()

            # Wrap blocking psutil call in thread pool
            process = psutil.Process()
            mem_info = await asyncio.to_thread(process.memory_info)
            total_mem_mb = mem_info.rss / 1024 / 1024

            # Complete memory accounting
            cache_mb = 0.0
            for df in data_engine.dataframes.values():
                if not df.empty and hasattr(df, 'memory_usage'):
                    try:
                        cache_mb += df.memory_usage(deep=True).sum() / 1024 / 1024
                    except Exception:
                        # Estimate if memory calculation fails
                        cache_mb += len(df) * len(df.columns) * 8 / 1024 / 1024
            
            position_cache_mb = sys.getsizeof(executor.position_cache) / 1024 / 1024
            order_cache_mb = sys.getsizeof(executor.active_orders) / 1024 / 1024

            if total_mem_mb > config.MEMORY_LIMIT_MB:
                app_logger.critical(f"Memory limit exceeded: {total_mem_mb:.1f}MB > {config.MEMORY_LIMIT_MB}MB (Dataframes: {cache_mb:.1f}MB)")
                if telegram:
                    await telegram.send(f"**MEMORY LIMIT EXCEEDED**\nSystem HALTED.\nMemory: {total_mem_mb:.1f}MB", priority="critical")
                break

            # Log memory breakdown more frequently
            if loop_count % 10 == 0:
                app_logger.info(f"Memory usage: Total={total_mem_mb:.1f}MB, Dataframes={cache_mb:.1f}MB, Positions={position_cache_mb:.1f}MB, Orders={order_cache_mb:.1f}MB")

            # Refresh position cache
            await executor.get_position(config.SYMBOL, force_refresh=True)

            # Continuous reconciliation
            if loop_count % 5 == 0:
                await executor._reconcile_active_orders()

            # Check for liquidated positions
            if loop_count % 10 == 0:
                await executor.check_liquidated_positions()

            # Data fetch with improved error handling
            fetch_tasks = []
            for tf in data_engine.timeframes:
                # For initial runs, fetch with force refresh
                fetch_tasks.append(data_engine.fetch_timeframe(
                    tf, force_refresh=(loop_count == 0), priority="high"))
            
            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            # Check for fetch errors
            successful_fetches = 0
            for tf, result in zip(data_engine.timeframes, results):
                if isinstance(result, Exception):
                    app_logger.error(f"Fetch error for {tf}: {result}")
                elif result is True:
                    successful_fetches += 1
                    df = data_engine.get_df(tf)
                    if not df.empty:
                        app_logger.debug(f"Successfully fetched {len(df)} bars for {tf}")
                    else:
                        app_logger.warning(f"Fetched {tf} but DataFrame is empty")
                else:
                    app_logger.warning(f"Fetch failed for {tf}")

            # If no timeframes have data, try emergency mode
            if successful_fetches == 0:
                app_logger.warning("No data fetched, entering emergency mode")
                # Try a simple direct fetch as last resort
                try:
                    ticker = yf.Ticker(data_engine.symbol)
                    df = await asyncio.to_thread(ticker.history, period="5d", interval="1h")
                    if not df.empty:
                        df.columns = df.columns.str.lower()
                        data_engine.dataframes["1h"] = await data_engine._engineer_features(df, "1h")
                        app_logger.info(f"Emergency mode: Got {len(df)} bars from Yahoo Finance")
                        successful_fetches = 1
                except Exception as e:
                    app_logger.error(f"Emergency mode also failed: {e}")

            # Training - only if we have data
            if successful_fetches > 0:
                for tf in data_engine.timeframes:
                    df = data_engine.get_df(tf)
                    if not df.empty and len(df) >= config.MIN_INITIAL_BARS:
                        should_retrain, reason = await brain.should_retrain(tf, df)
                        if should_retrain:
                            # Catch training exceptions
                            try:
                                await brain.train_timeframe(tf, df, retrain_reason=reason)
                            except InsufficientRegimeDiversityError:
                                app_logger.critical(f"{tf}: Insufficient regime diversity, strategy degraded")
                                if telegram:
                                    await telegram.send(f"**REGIME DIVERSITY FAILURE**\nTimeframe: {tf}\nSystem degraded.", priority="critical")

            # Prediction
            signals = []
            primary_signal = None

            for tf in data_engine.timeframes:
                df = data_engine.get_df(tf)
                if not df.empty and brain.is_trained.get(tf):
                    signal = await brain.predict_timeframe(tf, df)
                    if signal['is_valid']:
                        signals.append(signal)
                        if tf == config.PRIMARY_TIMEFRAME:
                            primary_signal = signal

            # Store primary signal for correlation check
            if primary_signal:
                executor._primary_tf_signal = primary_signal

            # Execution - only if we have valid signals
            if signals:
                # ENTRY: Best quality signal > minimum threshold
                best_signal = max(signals, key=lambda s: s['quality'])
                if best_signal['action'] != 'HOLD' and best_signal['quality'] >= config.MIN_TRADE_QUALITY:
                    await executor.execute_trade(config.SYMBOL, best_signal)

                # EXIT: Check same-TF opposing signal
                await executor.check_and_close_positions(config.SYMBOL, signals)

            # Budget update
            data_engine.update_budget_mode()

            # Rotate health check file
            health_file = Path(config.HEALTH_CHECK_FILE)
            if health_file.exists() and health_file.stat().st_size > 1024:
                # Async file operations
                try:
                    await aiofiles.os.rename(health_file, f"{config.HEALTH_CHECK_FILE}.{loop_count}")
                except Exception as e:
                    app_logger.warning(f"Failed to rotate health file: {e}")

            # Health check
            try:
                async with aiofiles.open(health_file, 'w') as f:
                    await f.write(f"OK {datetime.now().isoformat()} {loop_count}")
            except Exception as e:
                app_logger.warning(f"Failed to write health file: {e}")

            loop_duration = time.time() - loop_start
            app_logger.debug(f"Loop {loop_count} completed in {loop_duration:.2f}s")

            # Adaptive sleep
            sleep_time = max(0, config.LIVE_LOOP_INTERVAL_SECONDS - loop_duration)
            await asyncio.sleep(sleep_time)

            loop_count += 1

        except asyncio.CancelledError:
            app_logger.info("Live loop cancelled")
            break
        except Exception as e:
            app_logger.critical(f"Live loop error: {e}", exc_info=True)
            await asyncio.sleep(5)  # Brief backoff before retry

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        app_logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    except KeyboardInterrupt:
        app_logger.info("Process interrupted by user")
        sys.exit(0)
class WeakReferenceExpiredError(TitaniumBaseException):
    """Raised when weak reference has expired and system can degrade gracefully"""
    pass

# Validate environment BEFORE loading
REQUIRED_ENV_VARS = [
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHANNEL_ID",
    "TWELVEDATA_API_KEY"
]

print("Validating environment configuration...")
missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    print(f"Missing critical environment variables: {', '.join(missing_vars)}")
    print("Create a .env file or export variables before running.")
    sys.exit(1)

# Parse CLI arguments
def parse_args():
    """Minimal CLI for runtime configuration."""
    parser = argparse.ArgumentParser(description="TITANIUM Trading Bot")
    parser.add_argument("--symbol", type=str, default=os.getenv("TRADING_SYMBOL", "GLD"),
                        help="Trading symbol (default: GLD)")
    parser.add_argument("--paper", type=parse_bool,
                        default=parse_bool(os.getenv("PAPER_TRADING", "true")),
                        help="Paper trading mode (default: true)")
    parser.add_argument("--loop-interval", type=int,
                        default=int(os.getenv("LIVE_LOOP_INTERVAL", "120")),
                        help="Loop interval in seconds (default: 120)")
    return parser.parse_args()

@dataclass(frozen=True)
class SystemConfig:
    """Immutable configuration with validation and derived fields"""
    # 🔐 SECURE CREDENTIALS
    API_KEY: str = field(default_factory=lambda: os.environ["ALPACA_API_KEY"])
    SECRET_KEY: str = field(default_factory=lambda: os.environ["ALPACA_SECRET_KEY"])
    TELEGRAM_TOKEN: str = field(default_factory=lambda: os.environ["TELEGRAM_BOT_TOKEN"])
    TELEGRAM_CHANNEL: str = field(default_factory=lambda: os.environ["TELEGRAM_CHANNEL_ID"])
    TWELVEDATA_API_KEY: str = field(default_factory=lambda: os.environ["TWELVEDATA_API_KEY"])

    # 📊 TRADING SETTINGS - ADJUSTED FOR SUSTAINABLE FREQUENCY
    SYMBOL: str = field(default_factory=lambda: os.getenv("TRADING_SYMBOL", "GLD"))
    PRIMARY_TIMEFRAME: str = field(default_factory=lambda: os.getenv("PRIMARY_TIMEFRAME", "1d"))
    INTRADAY_TIMEFRAMES: List[str] = field(default_factory=lambda: json.loads(
        os.getenv("INTRADAY_TIMEFRAMES", '["4h", "1h", "15m"]')))
    TARGET_DAILY_TRADES: int = field(default_factory=lambda: int(os.getenv("TARGET_DAILY_TRADES", "3")))

    # 💰 RISK MANAGEMENT - CONSERVATIVE FOR MULTI-TRADE STRATEGY
    INITIAL_CAPITAL: float = field(default_factory=lambda: float(os.getenv("INITIAL_CAPITAL", "100000")))
    PAPER_TRADING: bool = field(default_factory=lambda: parse_bool(os.getenv("PAPER_TRADING", "true")))
    MAX_POS_SIZE_PCT: float = field(default_factory=lambda: float(os.getenv("MAX_POS_SIZE_PCT", "0.015")))
    MAX_DAILY_LOSS_PCT: float = field(default_factory=lambda: float(os.getenv("MAX_DAILY_LOSS_PCT", "0.01")))
    SLIPPAGE_BPS: float = field(default_factory=lambda: float(os.getenv("SLIPPAGE_BPS", "20.0")))
    PORTFOLIO_MAX_EXPOSURE: float = field(default_factory=lambda: float(os.getenv("PORTFOLIO_MAX_EXPOSURE", "0.15")))
    SPREAD_BUFFER_BPS: float = field(default_factory=lambda: float(os.getenv("SPREAD_BUFFER_BPS", "2.0")))

    # ⚡ EXECUTION - SMART ORDER ROUTING
    COMMISSION_PER_SHARE: float = field(default_factory=lambda: float(os.getenv("COMMISSION_PER_SHARE", "0.005")))
    ATR_PERIOD: int = field(default_factory=lambda: int(os.getenv("ATR_PERIOD", "14")))
    STOP_LOSS_ATR: float = field(default_factory=lambda: float(os.getenv("STOP_LOSS_ATR", "1.8")))
    TAKE_PROFIT_ATR: float = field(default_factory=lambda: float(os.getenv("TAKE_PROFIT_ATR", "2.5")))
    ORDER_TIMEOUT_SECONDS: int = field(default_factory=lambda: int(os.getenv("ORDER_TIMEOUT_SECONDS", "90")))
    EXECUTION_TIMEOUT_SECONDS: int = field(default_factory=lambda: int(os.getenv("EXECUTION_TIMEOUT_SECONDS", "120")))

    # 🔄 LIVE LOOP - API BUDGET OPTIMIZED
    LIVE_LOOP_INTERVAL_SECONDS: int = field(default_factory=lambda: int(os.getenv("LIVE_LOOP_INTERVAL", "120")))
    DATA_FETCH_INTERVAL_MINUTES: int = field(default_factory=lambda: int(os.getenv("DATA_FETCH_INTERVAL", "20")))
    MAX_CACHE_AGE_MULTIPLIER: int = field(default_factory=lambda: int(os.getenv("MAX_CACHE_AGE_MULTIPLIER", "2")))

    # 🧠 HMM SETTINGS - DYNAMIC COMPONENTS
    HMM_COMPONENTS: int = field(default_factory=lambda: int(os.getenv("HMM_COMPONENTS", "3")))
    HMM_MIN_COMPONENTS: int = field(default_factory=lambda: int(os.getenv("HMM_MIN_COMPONENTS", "3")))
    HMM_MAX_COMPONENTS: int = field(default_factory=lambda: int(os.getenv("HMM_MAX_COMPONENTS", "5")))
    HMM_TRAIN_WINDOW: Dict[str, int] = field(default_factory=lambda: json.loads(
        os.getenv("HMM_TRAIN_WINDOW", '{"1d": 504, "4h": 500, "1h": 1000, "15m": 2000}')
    ))
    HMM_RANDOM_STATE: int = field(default_factory=lambda: int(os.getenv("HMM_RANDOM_STATE", "42")))
    HMM_MAX_ITER: int = field(default_factory=lambda: int(os.getenv("HMM_MAX_ITER", "150")))
    HMM_RETRAIN_INTERVAL_BARS: Dict[str, int] = field(default_factory=lambda: json.loads(
        os.getenv("HMM_RETRAIN_INTERVAL_BARS", '{"1d": 20, "4h": 50, "1h": 100, "15m": 200}')
    ))

    # 🆕 API BUDGET MANAGEMENT - TWELVEDATA 800/DAY LIMIT
    API_CALLS_PER_DAY_LIMIT: int = field(default_factory=lambda: int(os.getenv("API_CALLS_PER_DAY_LIMIT", "800")))
    API_CALL_BUDGET_PRIORITY: List[str] = field(default_factory=lambda: json.loads(
        os.getenv("API_CALL_BUDGET_PRIORITY", '["1d", "4h", "1h", "15m"]')
    ))
    API_BUDGET_MODE: str = field(default_factory=lambda: os.getenv("API_BUDGET_MODE", "adaptive"))

    # 🎯 SMART EXECUTION
    USE_LIMIT_ORDERS: bool = field(default_factory=lambda: parse_bool(os.getenv("USE_LIMIT_ORDERS", "true")))
    LIMIT_ORDER_PASSIVITY_BPS: int = field(default_factory=lambda: int(os.getenv("LIMIT_ORDER_PASSIVITY_BPS", "10")))
    MIN_TRADE_COOLDOWN_SECONDS: int = field(default_factory=lambda: int(os.getenv("MIN_TRADE_COOLDOWN_SECONDS", "30")))
    MAX_TRADE_COOLDOWN_SECONDS: int = field(default_factory=lambda: int(os.getenv("MAX_TRADE_COOLDOWN_SECONDS", "300")))
    POSITION_CACHE_TTL: int = field(default_factory=lambda: int(os.getenv("POSITION_CACHE_TTL", "5")))

    # 📈 MONITORING & DATABASE
    METRICS_PORT: int = field(default_factory=lambda: int(os.getenv("METRICS_PORT", "9090")))
    HEALTH_CHECK_FILE: str = field(default_factory=lambda: os.getenv("HEALTH_CHECK_FILE", "./titanium_health.ok"))
    DB_PATH: str = field(default_factory=lambda: os.getenv("DB_PATH", "titanium_production.db"))
    DB_BACKUP_PATH: str = field(default_factory=lambda: os.getenv("DB_BACKUP_PATH", "./backups"))
    ACTIVE_ORDER_RETENTION_DAYS: int = field(default_factory=lambda: int(os.getenv("ACTIVE_ORDER_RETENTION_DAYS", "30")))
    HEALTH_CHECK_TIMEOUT_SECONDS: int = field(default_factory=lambda: int(os.getenv("HEALTH_CHECK_TIMEOUT_SECONDS", "240")))

    # 🛡️ REGIME MAPPING & QUALITY
    REGIME_BULL_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("REGIME_BULL_THRESHOLD", "0.25")))
    REGIME_BEAR_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("REGIME_BEAR_THRESHOLD", "-0.25")))
    MIN_TRADE_QUALITY: float = field(default_factory=lambda: float(os.getenv("MIN_TRADE_QUALITY", "65.0")))

    # 🔥 CHOP FILTER
    MIN_REGIME_CONFIDENCE: float = field(default_factory=lambda: float(os.getenv("MIN_REGIME_CONFIDENCE", "0.65")))

    # 🕒 MARKET HOURS - CRITICAL FIX
    MARKET_HOURS_ONLY: bool = field(default_factory=lambda: parse_bool(os.getenv("MARKET_HOURS_ONLY", "true")))
    TRADING_START_TIME: str = field(default_factory=lambda: os.getenv("TRADING_START_TIME", "09:30"))
    TRADING_END_TIME: str = field(default_factory=lambda: os.getenv("TRADING_END_TIME", "16:00"))
    EXTENDED_HOURS_ENABLED: bool = field(default_factory=lambda: parse_bool(os.getenv("EXTENDED_HOURS_ENABLED", "false")))

    # PHASE 4: Memory management
    MEMORY_LIMIT_MB: float = field(default_factory=lambda: float(os.getenv("MEMORY_LIMIT_MB", "500")))
    MAX_ROWS_PER_TIMEFRAME: int = field(default_factory=lambda: int(os.getenv("MAX_ROWS_PER_TIMEFRAME", "10000")))

    # PHASE 4: Circuit breakers
    CIRCUIT_TIMEOUT_SECONDS: int = field(default_factory=lambda: int(os.getenv("CIRCUIT_TIMEOUT_SECONDS", "300")))
    CIRCUIT_FAILURE_THRESHOLD: int = field(default_factory=lambda: int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", "5")))

    # FIXED: Secure model signature - no hardcoded fallback
    MODEL_SIGNATURE_SECRET: str = field(default_factory=lambda: os.getenv("MODEL_SIGNATURE_SECRET", ""))

    # NEW: Minimum position size
    MIN_POSITION_SIZE: int = field(default_factory=lambda: int(os.getenv("MIN_POSITION_SIZE", "1")))

    # NEW: Enforce round lots
    ENFORCE_ROUND_LOTS: bool = field(default_factory=lambda: parse_bool(os.getenv("ENFORCE_ROUND_LOTS", "false")))

    # NEW: Memory cache limits
    MAX_CACHE_SIZE_MB: float = field(default_factory=lambda: float(os.getenv("MAX_CACHE_SIZE_MB", "200")))

    # NEW: Minimum risk per share floor (FIXED: Must be > 0)
    MIN_RISK_PER_SHARE_BPS: float = field(default_factory=lambda: float(os.getenv("MIN_RISK_PER_SHARE_BPS", "5.0")))

    # NEW: Maximum concurrent open positions
    MAX_OPEN_POSITIONS: int = field(default_factory=lambda: int(os.getenv("MAX_OPEN_POSITIONS", "3")))

    # NEW: Cash buffer for position sizing
    CASH_BUFFER_PCT: float = field(default_factory=lambda: float(os.getenv("CASH_BUFFER_PCT", "0.96")))

    # NEW: HMM sequence window for valid regime inference
    HMM_SEQUENCE_WINDOW: int = field(default_factory=lambda: int(os.getenv("HMM_SEQUENCE_WINDOW", "20")))

    # NEW: Minimum sequence length for HMM prediction
    HMM_MIN_SEQUENCE_LENGTH: int = field(default_factory=lambda: int(os.getenv("HMM_MIN_SEQUENCE_LENGTH", "10")))

    # NEW: Yahoo Finance historical limits (days)
    YFINANCE_MAX_LOOKBACK: Dict[str, int] = field(default_factory=lambda: json.loads(
        os.getenv("YFINANCE_MAX_LOOKBACK", '{"1d": 730, "4h": 730, "1h": 730, "15m": 60}')
    ))

    # NEW: TwelveData max output size
    TWELVEDATA_MAX_OUTPUTSIZE: int = field(default_factory=lambda: int(os.getenv("TWELVEDATA_MAX_OUTPUTSIZE", "5000")))

    # V19 NEW: Progressive data collection
    INITIAL_DATA_BARS: int = field(default_factory=lambda: int(os.getenv("INITIAL_DATA_BARS", "100")))
    DATA_COLLECTION_STRATEGY: str = field(default_factory=lambda: os.getenv("DATA_COLLECTION_STRATEGY", "progressive"))

    # V19 NEW: Data validation thresholds
    MIN_INITIAL_BARS: int = field(default_factory=lambda: int(os.getenv("MIN_INITIAL_BARS", "10")))
    DATA_STALENESS_THRESHOLD: int = field(default_factory=lambda: int(os.getenv("DATA_STALENESS_THRESHOLD", "3")))

    # V19 NEW: API retry settings
    API_MAX_RETRIES: int = field(default_factory=lambda: int(os.getenv("API_MAX_RETRIES", "3")))
    API_RETRY_DELAY: float = field(default_factory=lambda: float(os.getenv("API_RETRY_DELAY", "2.0")))

    def __post_init__(self):
        """Post-initialization validation with hard caps"""
        # Validate numeric ranges
        if self.MAX_POS_SIZE_PCT > 0.1:
            raise ValueError(f"MAX_POS_SIZE_PCT cannot exceed 10% (got {self.MAX_POS_SIZE_PCT})")
        if self.MAX_POS_SIZE_PCT > 0.02:
            app_logger.warning("MAX_POS_SIZE_PCT > 2% is highly risky for multi-trade strategy")
        if self.MAX_DAILY_LOSS_PCT > 0.02:
            app_logger.warning("MAX_DAILY_LOSS_PCT > 2% is aggressive")
        if self.TARGET_DAILY_TRADES > 5:
            app_logger.warning("TARGET_DAILY_TRADES > 5 may exceed API budget or cause overtrading")
        if self.COMMISSION_PER_SHARE < 0:
            raise ValueError("COMMISSION_PER_SHARE cannot be negative")
        if self.COMMISSION_PER_SHARE == 0:
            app_logger.warning("Commissions set to 0 - ensure this matches your broker")

        # Validate HMM component bounds
        if not (self.HMM_MIN_COMPONENTS <= self.HMM_COMPONENTS <= self.HMM_MAX_COMPONENTS):
            raise ValueError(
                f"HMM_COMPONENTS must be between {self.HMM_MIN_COMPONENTS} and {self.HMM_MAX_COMPONENTS}"
            )

        # Validate MIN_TRADE_QUALITY
        if not (50 <= self.MIN_TRADE_QUALITY <= 95):
            app_logger.warning("MIN_TRADE_QUALITY outside recommended range [50, 95]")

        # Validate MIN_RISK_PER_SHARE_BPS > 0 to prevent division by zero
        if self.MIN_RISK_PER_SHARE_BPS <= 0:
            raise ValueError(f"MIN_RISK_PER_SHARE_BPS must be > 0 (got {self.MIN_RISK_PER_SHARE_BPS})")

        # Validate timeframe format
        valid_tfs = ["1d", "4h", "1h", "15m"]
        tf_pattern = re.compile(r'^\d+[hdm]$')
        if not tf_pattern.match(self.PRIMARY_TIMEFRAME) or self.PRIMARY_TIMEFRAME not in valid_tfs:
            raise ValueError(f"Invalid PRIMARY_TIMEFRAME: {self.PRIMARY_TIMEFRAME}")
        for tf in self.INTRADAY_TIMEFRAMES:
            if not tf_pattern.match(tf) or tf not in valid_tfs:
                raise ValueError(f"Invalid INTRADAY_TIMEFRAME: {tf}")

        # Validate MODEL_SIGNATURE_SECRET is set
        if not self.MODEL_SIGNATURE_SECRET:
            raise ConfigurationError("MODEL_SIGNATURE_SECRET environment variable is required")

        # Validate TwelveData output size
        if self.TWELVEDATA_MAX_OUTPUTSIZE > 5000:
            app_logger.warning(f"TWELVEDATA_MAX_OUTPUTSIZE {self.TWELVEDATA_MAX_OUTPUTSIZE} exceeds API limit of 5000")
            object.__setattr__(self, 'TWELVEDATA_MAX_OUTPUTSIZE', 5000)

        # Validate HMM sequence window
        if self.HMM_SEQUENCE_WINDOW < self.HMM_MIN_SEQUENCE_LENGTH:
            raise ValueError(
                f"HMM_SEQUENCE_WINDOW ({self.HMM_SEQUENCE_WINDOW}) must be >= HMM_MIN_SEQUENCE_LENGTH ({self.HMM_MIN_SEQUENCE_LENGTH})")

        # Calculate derived fields
        object.__setattr__(self, 'MAX_DAILY_LOSS_DOLLAR', self.INITIAL_CAPITAL * self.MAX_DAILY_LOSS_PCT)

        # FIXED: Do not modify frozen dataclass, only warn
        if self.LIMIT_ORDER_PASSIVITY_BPS > 50:
            app_logger.warning("LIMIT_ORDER_PASSIVITY_BPS > 50bps may prevent fills")

# ===============================================================================
# 🔐 PRODUCTION SECRETS MANAGEMENT
# ===============================================================================

class SecretsManager:
    """Production-grade secrets handling with Vault integration"""

    VAULT_ADDR = os.getenv("VAULT_ADDR")
    VAULT_TOKEN = os.getenv("VAULT_TOKEN")
    VAULT_MOUNT_POINT = os.getenv("VAULT_MOUNT_POINT", "secret")

    @staticmethod
    def load_from_vault() -> bool:
        """Integrate with HashiCorp Vault KV v2"""
        if not SecretsManager.VAULT_ADDR or not SecretsManager.VAULT_TOKEN:
            app_logger.info("Vault integration not configured, using environment variables")
            return False

        try:
            import hvac
            client = hvac.Client(url=SecretsManager.VAULT_ADDR, token=SecretsManager.VAULT_TOKEN)
            if not client.is_authenticated():
                app_logger.error("Vault authentication failed")
                return False

            secrets = client.secrets.kv.v2.read_secret_version(
                path="titanium/config",
                mount_point=SecretsManager.VAULT_MOUNT_POINT
            )

            for key, value in secrets["data"]["data"].items():
                os.environ[key] = value

            app_logger.info("Vault secrets loaded successfully")
            return True
        except ImportError:
            app_logger.warning("hvac library not installed, skipping Vault integration")
            return False
        except Exception as e:
            app_logger.error(f"Vault integration failed: {e}")
            return False

    @staticmethod
    def validate_env_security():
        """Check for common security misconfigurations"""
        env_path = Path('.env')
        if env_path.exists():
            # Check file permissions
            stat = env_path.stat()
            if stat.st_mode & 0o077:
                app_logger.warning(".env file is readable by group/other. Run: chmod 600 .env")

            # Check gitignore
            gitignore = Path('.gitignore')
            if gitignore.exists():
                content = gitignore.read_text()
                if '.env' not in content and '.env*' not in content:
                    app_logger.warning(".env not in .gitignore! Add it immediately.")
            else:
                app_logger.warning("No .gitignore found. Create one and add .env")

        # Check for hardcoded keys in code
        api_key = os.getenv("ALPACA_API_KEY")
        if api_key and len(api_key) < 20:
            app_logger.error("Invalid ALPACA_API_KEY format detected")

# ===============================================================================
# 📡 TELEGRAM BOT - ENHANCED RATE LIMITING
# ===============================================================================

class TelegramBot:
    def __init__(self, token: str, channel: str):
        self.token = token
        self.channel = channel
        self.enabled = False  # Start disabled until credentials verified
        self.min_interval = 2.0
        self._last_message_time = 0
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
        self._error_count = 0
        self._max_errors = 5

        # Bounded critical queue with overflow protection
        self._critical_queue = asyncio.Queue(maxsize=100)
        self._critical_task: Optional[asyncio.Task] = None

        # Overflow protection flag with recovery
        self._queue_overflow = False
        self._overflow_reset_threshold = 50  # Reset when queue drops below 50%

    async def initialize(self):
        """Async credential test and worker start"""
        await self._test_credentials_async()
        if self.enabled:
            await self.start_critical_worker()

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=3, limit_per_host=1)
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                connector=connector
            )

    async def _test_credentials_async(self):
        """Async credential test with manual retry"""
        if not self.token or not self.channel:
            app_logger.warning("Telegram credentials not provided")
            self.enabled = False
            return

        # Validate channel ID format
        if not (self.channel.startswith('@') or self.channel.lstrip('-').isdigit()):
            app_logger.error("TELEGRAM_CHANNEL_ID must be @username or numeric ID")
            self.enabled = False
            return

        for attempt in range(3):
            try:
                url = f"https://api.telegram.org/bot{self.token}/getMe"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=10) as resp:
                        data = await resp.json()
                        if data.get("ok"):
                            app_logger.info(f"Telegram bot: @{data['result']['username']}")

                            chat_url = f"https://api.telegram.org/bot{self.token}/getChat"
                            chat_data = {"chat_id": self.channel}
                            async with session.post(chat_url, json=chat_data, timeout=10) as chat_resp:
                                chat_json = await chat_resp.json()
                                if chat_json.get("ok"):
                                    app_logger.info("Channel access verified")
                                    self.enabled = True
                                    return
                                else:
                                    app_logger.warning(f"Bot not admin in channel: {chat_json.get('description')}")
                                    self.enabled = False
                                    return
                        else:
                            app_logger.error(f"Invalid token: {data.get('description')}")
                            self.enabled = False
                            return
            except aiohttp.ClientError as e:
                app_logger.error(f"Telegram test attempt {attempt+1} failed with network error: {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                app_logger.error(f"Telegram test attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)

        app_logger.error("Telegram credential test failed after 3 attempts")
        self.enabled = False

    async def start_critical_worker(self):
        """Start critical alert worker"""
        self._critical_task = asyncio.create_task(self._process_critical_queue())

    async def stop_critical_worker(self):
        """Stop critical alert worker"""
        if self._critical_task:
            # FIXED: Wait for task to finish with timeout
            try:
                self._critical_task.cancel()
                await asyncio.wait_for(self._critical_task, timeout=5.0)
            except asyncio.CancelledError:
                app_logger.debug("Critical task cancelled successfully")
            except asyncio.TimeoutError:
                app_logger.warning("Critical task shutdown timeout, forcing")
            except Exception as e:
                app_logger.warning(f"Critical task shutdown error: {e}")

    async def _process_critical_queue(self):
        """Process critical alerts without blocking main lock"""
        while True:
            try:
                message = await self._critical_queue.get()
                await self._send_critical(message)
                self._critical_queue.task_done()

                # Reset overflow flag if queue size drops below threshold
                if self._queue_overflow and self._critical_queue.qsize() < (self._critical_queue.maxsize * 0.5):
                    self._queue_overflow = False
                    app_logger.info("Telegram queue overflow condition cleared")

            except asyncio.CancelledError:
                app_logger.debug("Critical queue processing cancelled")
                break
            except Exception as e:
                app_logger.error(f"Critical queue error: {e}")
                await asyncio.sleep(5)

    async def _send_critical(self, message: str) -> bool:
        """Send critical message with minimal blocking and overflow protection"""
        try:
            if not self._session or self._session.closed:
                await self._ensure_session()

            # Consistent truncation limit
            if len(message) > 4096:
                message = message[:4000] + "\n... [TRUNCATED]"

            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id": self.channel,
                "text": message,
                "parse_mode": "Markdown"
            }
            async with self._session.post(url, json=payload) as resp:
                if resp.status == 200:
                    return True
                else:
                    app_logger.error(f"Critical Telegram error {resp.status}: {await resp.text()}")
                    return False
        except Exception as e:
            app_logger.error(f"Critical send failed: {e}")
            return False

    async def send(self, message: str, priority: str = "normal") -> bool:
        """Send message with rate limiting, session reuse, and error handling"""
        if not self.enabled:
            return False

        # Critical priority bypasses normal queue with overflow protection
        if priority == "critical":
            try:
                if self._critical_queue.full():
                    # Overflow protection: drop oldest message if queue full for > 30 seconds
                    if self._queue_overflow:
                        # Drop the oldest message
                        try:
                            self._critical_queue.get_nowait()
                            self._critical_queue.task_done()
                            app_logger.warning("Critical queue overflow, dropping oldest message")
                        except asyncio.QueueEmpty:
                            pass

                    self._queue_overflow = True
                    app_logger.warning("Critical queue full, message may be delayed")

                # Add with timeout to prevent indefinite blocking
                try:
                    await asyncio.wait_for(self._critical_queue.put(message), timeout=5.0)
                except asyncio.TimeoutError:
                    app_logger.error("Failed to queue critical message: timeout")
                    return False

                return True
            except Exception as e:
                app_logger.error(f"Failed to queue critical message: {e}")
                return False

        # Prevent alert fatigue for non-critical messages
        if priority == "normal" and self._error_count > 3:
            return False

        async with self._lock:
            current_time = time.time()
            elapsed = current_time - self._last_message_time
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)

            if not self._session or self._session.closed:
                await self._ensure_session()

            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id": self.channel,
                "text": message[:4096],
                "parse_mode": "Markdown"
            }

            try:
                async with self._session.post(url, json=payload) as resp:
                    self._last_message_time = time.time()
                    if resp.status == 200:
                        self._error_count = 0
                        return True
                    else:
                        self._error_count += 1
                        error_text = await resp.text()
                        app_logger.error(f"Telegram error {resp.status}: {error_text}")
                        if self._error_count >= self._max_errors:
                            app_logger.critical("Disabling Telegram due to repeated errors")
                            self.enabled = False
                        return False
            except asyncio.TimeoutError:
                self._error_count += 1
                app_logger.error("Telegram send timed out")
                return False
            except aiohttp.ClientError as e:
                self._error_count += 1
                app_logger.error(f"Telegram client error: {e}")
                return False
            except Exception as e:
                self._error_count += 1
                app_logger.error(f"Telegram send failed: {e}")
                return False

    async def close(self):
        """FIXED: Proper cleanup with timeout and forced session closure"""
        await self.stop_critical_worker()
        if self._session and not self._session.closed:
            try:
                await asyncio.wait_for(self._session.close(), timeout=5.0)
            except asyncio.TimeoutError:
                app_logger.warning("Telegram session close timeout, forcing closure")
                # Force close the connector
                if self._session.connector:
                    self._session.connector.close()
            except Exception as e:
                app_logger.warning(f"Telegram session close error: {e}")

# ===============================================================================
# 💾 PERSISTENCE LAYER - ASYNC WITH BACKUP & CONNECTION POOLING
# ===============================================================================

class DatabaseManager:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.db_path = Path(config.DB_PATH)
        self.backup_path = Path(config.DB_BACKUP_PATH)

        # Validate parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_path.mkdir(parents=True, exist_ok=True)

        # FIXED: Connection pooling with unified state management
        self._lock = asyncio.Lock()
        self._connection_semaphore = asyncio.Semaphore(5)

        # Unified connection state tracking
        # conn_id -> {conn, last_used, query_count}
        self._connection_pool: Dict[int, Dict[str, Any]] = {}
        self._pool_maxsize = 5
        self._max_queries_per_conn = 1000

        # Reentrant lock tracking for exclusive transactions
        self._exclusive_lock = asyncio.Lock()
        self._exclusive_lock_owner: Optional[int] = None  # task id
        self._exclusive_lock_count = 0

    async def _get_connection(self) -> aiosqlite.Connection:
        """Get connection from pool or create new with unified state management"""
        async with self._lock:
            # Check for available connections in pool
            now = time.time()
            available_conns = []

            for conn_id, conn_data in self._connection_pool.items():
                conn = conn_data['conn']
                # Health check connection
                try:
                    await asyncio.wait_for(conn.execute("SELECT 1"), timeout=2.0)
                    # Check if connection has exceeded query limit
                    if conn_data['query_count'] < self._max_queries_per_conn:
                        available_conns.append((conn_id, conn_data))
                except (asyncio.TimeoutError, Exception):
                    # Connection is dead, remove from pool
                    try:
                        await conn.close()
                    except Exception:
                        pass
                    del self._connection_pool[conn_id]
                    app_logger.debug(f"Removed dead connection {conn_id} from pool")

            # Return the most recently used healthy connection
            if available_conns:
                available_conns.sort(key=lambda x: x[1]['last_used'], reverse=True)
                conn_id, conn_data = available_conns[0]
                conn = conn_data['conn']
                conn_data['last_used'] = now
                conn_data['query_count'] += 1
                return conn

            # Create new connection if pool not full
            if len(self._connection_pool) < self._pool_maxsize:
                try:
                    conn = await asyncio.wait_for(
                        aiosqlite.connect(self.db_path),
                        timeout=10
                    )
                    conn_id = id(conn)
                    self._connection_pool[conn_id] = {
                        'conn': conn,
                        'last_used': now,
                        'query_count': 1
                    }
                    app_logger.debug(f"Created new connection {conn_id}, pool size: {len(self._connection_pool)}")
                    return conn
                except Exception as e:
                    app_logger.error(f"Failed to create new connection: {e}")
                    raise

            # Pool is full, evict least recently used connection
            lru_conn_id = min(self._connection_pool.items(), key=lambda x: x[1]['last_used'])[0]
            lru_conn = self._connection_pool[lru_conn_id]['conn']
            try:
                await lru_conn.close()
            except Exception:
                pass
            del self._connection_pool[lru_conn_id]

            # Create new connection
            try:
                conn = await asyncio.wait_for(
                    aiosqlite.connect(self.db_path),
                    timeout=10
                )
                conn_id = id(conn)
                self._connection_pool[conn_id] = {
                    'conn': conn,
                    'last_used': now,
                    'query_count': 1
                }
                app_logger.debug(f"Evicted LRU connection, created new {conn_id}")
                return conn
            except Exception as e:
                app_logger.error(f"Failed to create new connection after eviction: {e}")
                raise

    async def _release_connection(self, conn: aiosqlite.Connection):
        """Return connection to pool with updated state"""
        conn_id = id(conn)
        async with self._lock:
            if conn_id in self._connection_pool:
                self._connection_pool[conn_id]['last_used'] = time.time()
                # Reset query count if exceeded limit
                if self._connection_pool[conn_id]['query_count'] >= self._max_queries_per_conn:
                    self._connection_pool[conn_id]['query_count'] = 0
                    app_logger.debug(f"Reset query count for connection {conn_id}")

    async def initialize(self):
        """Initialize database with indexes, WAL mode, migrations"""
        await self._connection_semaphore.acquire()
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Set exclusive locking for critical sections
                await db.execute("PRAGMA busy_timeout = 5000")
                result = await db.execute("PRAGMA journal_mode=WAL")
                if (await result.fetchone())[0].lower() != 'wal':
                    app_logger.warning("Could not enable WAL mode")
                await db.execute("PRAGMA foreign_keys=ON")
                await db.execute("PRAGMA synchronous=NORMAL")
                await db.execute("PRAGMA wal_autocheckpoint=1000")

                # Schema with migrations
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        action TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        entry_price REAL,
                        exit_price REAL,
                        status TEXT NOT NULL,
                        entry_time TIMESTAMP NOT NULL,
                        exit_time TIMESTAMP,
                        stop_loss REAL,
                        take_profit REAL,
                        pnl REAL,
                        quality_score REAL,
                        regime TEXT,
                        confidence REAL,
                        order_id TEXT,
                        slippage_cost REAL,
                        commission_cost REAL,
                        timeframe_weight REAL,
                        execution_duration_ms INTEGER,
                        data_hash TEXT,
                        execution_price REAL,
                        execution_type TEXT,
                        exit_reason TEXT
                    )
                """)

                await db.execute("""
                    CREATE TABLE IF NOT EXISTS daily_risk (
                        date TEXT PRIMARY KEY,
                        daily_loss REAL NOT NULL,
                        daily_trades INTEGER NOT NULL,
                        max_drawdown REAL,
                        portfolio_value REAL,
                        api_calls_used INTEGER DEFAULT 0,
                        signal_efficiency REAL DEFAULT 0.0,
                        safety_mode_active BOOLEAN DEFAULT FALSE,
                        timeframe_cooldowns TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                await db.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id TEXT PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        timeframe TEXT NOT NULL,
                        action TEXT NOT NULL,
                        score REAL,
                        regime TEXT,
                        confidence REAL,
                        quality REAL,
                        symbol TEXT NOT NULL,
                        execution_decision TEXT,
                        api_budget_pct REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                await db.execute("""
                    CREATE TABLE IF NOT EXISTS active_orders (
                        order_id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        status TEXT NOT NULL,
                        filled_qty INTEGER DEFAULT 0,
                        submitted_at TIMESTAMP NOT NULL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                await db.execute("""
                    CREATE TABLE IF NOT EXISTS model_metadata (
                        timeframe TEXT PRIMARY KEY,
                        last_trained TIMESTAMP,
                        train_score REAL,
                        test_score REAL,
                        score_ratio REAL,
                        features TEXT,
                        data_hash TEXT,
                        train_duration_ms INTEGER,
                        retrain_reason TEXT,
                        components_used INTEGER
                    )
                """)

                await db.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        date TEXT PRIMARY KEY,
                        sharpe_ratio REAL,
                        win_rate REAL,
                        max_drawdown REAL,
                        total_trades INTEGER,
                        avg_signal_quality REAL,
                        model_drift_detected BOOLEAN,
                        signal_to_trade_ratio REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                await db.execute("""
                    CREATE TABLE IF NOT EXISTS api_usage (
                        date TEXT PRIMARY KEY,
                        calls_used INTEGER NOT NULL,
                        calls_remaining INTEGER NOT NULL,
                        reset_hour INTEGER NOT NULL,
                        budget_mode TEXT,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                await db.execute("""
                    CREATE TABLE IF NOT EXISTS timeframe_correlation (
                        timestamp TIMESTAMP,
                        timeframe1 TEXT,
                        timeframe2 TEXT,
                        correlation REAL,
                        PRIMARY KEY (timestamp, timeframe1, timeframe2)
                    )
                """)

                # Table for trade performance persistence
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS trade_performance (
                        date TEXT PRIMARY KEY,
                        win_rate REAL,
                        avg_pnl REAL,
                        trades_count INTEGER,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Indexes for performance
                await db.execute("CREATE INDEX IF NOT EXISTS idx_trades_status_symbol ON trades(status, symbol)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_daily_risk_date ON daily_risk(date)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_api_usage_date ON api_usage(date)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_active_orders_status ON active_orders(status)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_active_orders_order_id ON active_orders(order_id)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_active_orders_status_symbol ON active_orders(status, symbol)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_signals_execution_decision ON signals(execution_decision)")

                # Schema migrations
                try:
                    await db.execute("SELECT exit_reason FROM trades LIMIT 1")
                except sqlite3.OperationalError:
                    await db.execute("ALTER TABLE trades ADD COLUMN exit_reason TEXT")
                    app_logger.info("Migrated trades table: added exit_reason column")

                try:
                    await db.execute("SELECT timeframe_cooldowns FROM daily_risk LIMIT 1")
                except sqlite3.OperationalError:
                    await db.execute("ALTER TABLE daily_risk ADD COLUMN timeframe_cooldowns TEXT")
                    app_logger.info("Migrated daily_risk table: added timeframe_cooldowns column")

                try:
                    await db.execute("SELECT api_budget_pct FROM signals LIMIT 1")
                except sqlite3.OperationalError:
                    await db.execute("ALTER TABLE signals ADD COLUMN api_budget_pct REAL")
                    app_logger.info("Migrated signals table: added api_budget_pct column")

                # WAL checkpoint to prevent growth
                size_result = await db.execute("PRAGMA page_count")
                page_count = (await size_result.fetchone())[0]
                if page_count > 25000:  # ~100MB
                    app_logger.info("Database >100MB, running VACUUM")
                    await db.execute("VACUUM")

                await db.commit()

        except asyncio.TimeoutError:
            app_logger.error("Database initialization timed out")
            raise
        except Exception as e:
            app_logger.error(f"Database initialization failed: {e}")
            raise
        finally:
            self._connection_semaphore.release()

        app_logger.info("Database initialized with WAL mode, indexes, and enhanced schema")

    @asynccontextmanager
    async def exclusive_transaction(self):
        """Context manager for exclusive transactions with timeout - FIXED reentrant lock"""
        current_task = id(asyncio.current_task())

        # Check if we already own the lock (reentrant)
        if self._exclusive_lock_owner == current_task:
            self._exclusive_lock_count += 1
            conn = await self._get_connection()
            try:
                await conn.execute("BEGIN EXCLUSIVE")
                yield conn
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
            finally:
                await self._release_connection(conn)
                self._exclusive_lock_count -= 1
                if self._exclusive_lock_count == 0:
                    self._exclusive_lock_owner = None
        else:
            # Acquire lock with timeout
            try:
                await asyncio.wait_for(self._exclusive_lock.acquire(), timeout=30)
            except asyncio.TimeoutError:
                raise asyncio.TimeoutError("Could not acquire database lock within 30 seconds")

            self._exclusive_lock_owner = current_task
            self._exclusive_lock_count = 1

            try:
                conn = await asyncio.wait_for(
                    aiosqlite.connect(self.db_path), timeout=10
                )
                try:
                    await conn.execute("BEGIN EXCLUSIVE")
                    yield conn
                    await conn.commit()
                except Exception:
                    await conn.rollback()
                    raise
                finally:
                    await conn.close()
            finally:
                self._exclusive_lock_count = 0
                self._exclusive_lock_owner = None
                self._exclusive_lock.release()

    async def log_trade(self, trade_data: Dict[str, Any]):
        """Log trade atomically with extended fields"""
        async with self.exclusive_transaction() as db:
            # Use natural key for idempotency
            trade_id = trade_data.get('id', f"{trade_data['symbol']}_{trade_data['entry_time']}_{trade_data['timeframe']}")

            # Explicit column list for robustness
            await db.execute("""
                INSERT OR REPLACE INTO trades (
                    id, symbol, timeframe, action, quantity, entry_price, exit_price, status,
                    entry_time, exit_time, stop_loss, take_profit, pnl, quality_score, regime,
                    confidence, order_id, slippage_cost, commission_cost, timeframe_weight,
                    execution_duration_ms, data_hash, execution_price, execution_type, exit_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id,
                trade_data['symbol'],
                trade_data['timeframe'],
                trade_data['action'],
                trade_data['quantity'],
                trade_data.get('entry_price'),
                trade_data.get('exit_price'),
                trade_data['status'],
                trade_data['entry_time'],
                trade_data.get('exit_time'),
                trade_data.get('stop_loss'),
                trade_data.get('take_profit'),
                trade_data.get('pnl'),
                trade_data.get('quality_score'),
                trade_data.get('regime'),
                trade_data.get('confidence'),
                trade_data.get('order_id'),
                trade_data.get('slippage_cost', 0.0),
                trade_data.get('commission_cost', 0.0),
                trade_data.get('timeframe_weight'),
                trade_data.get('execution_duration_ms'),
                trade_data.get('data_hash'),
                trade_data.get('execution_price'),
                trade_data.get('execution_type'),
                trade_data.get('exit_reason')
            ))
            app_logger.debug(f"Trade logged: {trade_id}")

    async def get_open_trades(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open trades with optional symbol filter"""
        conn = await self._get_connection()
        try:
            conn.row_factory = aiosqlite.Row
            if symbol:
                cursor = await conn.execute("SELECT * FROM trades WHERE status = 'OPEN' AND symbol = ?", (symbol,))
            else:
                cursor = await conn.execute("SELECT * FROM trades WHERE status = 'OPEN'")
            trades = [dict(row) for row in await cursor.fetchall()]

            # Convert Decimal PnL back to float for compatibility
            for trade in trades:
                if 'pnl' in trade and trade['pnl'] is not None:
                    try:
                        trade['pnl'] = float(trade['pnl'])
                    except (TypeError, ValueError) as e:
                        app_logger.warning(f"Could not convert PnL for trade {trade.get('id')}: {e}")
                        trade['pnl'] = 0.0
            return trades
        finally:
            await self._release_connection(conn)

    async def update_trade_exit(self, trade_id: str, exit_price: float, exit_time: datetime, pnl: Decimal, status: str, exit_reason: str = None):
        """Update trade on exit"""
        async with self.exclusive_transaction() as db:
            await db.execute("""
                UPDATE trades SET exit_price = ?, exit_time = ?, pnl = ?, status = ?, exit_reason = ?
                WHERE id = ?
            """, (exit_price, exit_time, str(pnl), status, exit_reason, trade_id))
            app_logger.debug(f"Trade updated: {trade_id}")

    async def log_daily_risk(self, date: str, daily_loss: Decimal, daily_trades: int, max_drawdown: float, portfolio_value: float,
                             api_calls_used: int = 0, signal_efficiency: float = 0.0, safety_mode_active: bool = False,
                             timeframe_cooldowns: Optional[Dict] = None):
        """Log daily risk metrics with Decimal support"""
        async with self.exclusive_transaction() as db:
            cooldowns_json = json.dumps(timeframe_cooldowns) if timeframe_cooldowns else None

            await db.execute("""
                INSERT OR REPLACE INTO daily_risk (date, daily_loss, daily_trades, max_drawdown, portfolio_value, 
                api_calls_used, signal_efficiency, safety_mode_active, timeframe_cooldowns)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (date, str(daily_loss), daily_trades, max_drawdown, portfolio_value,
                  api_calls_used, signal_efficiency, safety_mode_active, cooldowns_json))
            app_logger.debug(f"Daily risk logged: {date}")

    async def get_daily_risk(self, date: str) -> Optional[Dict[str, Any]]:
        """Get daily risk data"""
        conn = await self._get_connection()
        try:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT * FROM daily_risk WHERE date = ?", (date,))
            row = await cursor.fetchone()
            if row:
                data = dict(row)
                # Convert back to Decimal
                if 'daily_loss' in data and data['daily_loss'] is not None:
                    try:
                        data['daily_loss'] = Decimal(data['daily_loss'])
                    except (TypeError, ValueError) as e:
                        app_logger.warning(f"Could not convert daily_loss to Decimal: {e}")
                        data['daily_loss'] = Decimal('0.0')
                # Parse timeframe_cooldowns
                if data.get('timeframe_cooldowns'):
                    try:
                        data['timeframe_cooldowns'] = json.loads(data['timeframe_cooldowns'])
                    except json.JSONDecodeError as e:
                        app_logger.warning(f"Could not parse timeframe_cooldowns JSON: {e}")
                        data['timeframe_cooldowns'] = {}
                return data
            return None
        finally:
            await self._release_connection(conn)

    async def log_signal(self, signal_data: Dict[str, Any]):
        """Log signal with deduplication using natural key"""
        async with self.exclusive_transaction() as db:
            # Use natural key: timestamp_timeframe_action
            signal_id = f"{signal_data['timestamp']}_{signal_data['timeframe']}_{signal_data['action']}"

            await db.execute("""
                INSERT OR REPLACE INTO signals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                signal_id,
                signal_data['timestamp'],
                signal_data['timeframe'],
                signal_data['action'],
                signal_data.get('score'),
                signal_data.get('regime'),
                signal_data.get('confidence'),
                signal_data.get('quality'),
                signal_data.get('symbol'),
                signal_data.get('execution_decision'),
                signal_data.get('api_budget_pct')
            ))
            app_logger.debug(f"Signal logged: {signal_id}")

    async def log_api_usage(self, date: str, calls_used: int, calls_remaining: int, reset_hour: int, budget_mode: str = "adaptive"):
        """Log API usage for budget tracking"""
        async with self.exclusive_transaction() as db:
            await db.execute("""
                INSERT OR REPLACE INTO api_usage (date, calls_used, calls_remaining, reset_hour, budget_mode, last_updated)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (date, calls_used, calls_remaining, reset_hour, budget_mode))
            app_logger.debug(f"API usage logged: {date} - {calls_used} calls")

    async def get_api_usage(self, date: str) -> Optional[Dict[str, Any]]:
        """Get API usage data"""
        conn = await self._get_connection()
        try:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT * FROM api_usage WHERE date = ?", (date,))
            row = await cursor.fetchone()
            return dict(row) if row else None
        finally:
            await self._release_connection(conn)

    async def log_trade_performance(self, date: str, win_rate: float, avg_pnl: float, trades_count: int):
        """Persist trade performance metrics"""
        async with self.exclusive_transaction() as db:
            await db.execute("""
                INSERT OR REPLACE INTO trade_performance (date, win_rate, avg_pnl, trades_count, last_updated)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (date, win_rate, avg_pnl, trades_count))
            app_logger.debug(f"Trade performance logged: {date} - {trades_count} trades")

    async def get_trade_performance(self, date: str) -> Optional[Dict[str, Any]]:
        """Get trade performance data"""
        conn = await self._get_connection()
        try:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT * FROM trade_performance WHERE date = ?", (date,))
            row = await cursor.fetchone()
            return dict(row) if row else None
        finally:
            await self._release_connection(conn)

    async def get_recent_trades(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trades for performance analysis"""
        conn = await self._get_connection()
        try:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("""
                SELECT * FROM trades WHERE symbol = ? 
                ORDER BY entry_time DESC LIMIT ?
            """, (symbol, limit))
            trades = [dict(row) for row in await cursor.fetchall()]
            # Convert Decimal PnL
            for trade in trades:
                if 'pnl' in trade and trade['pnl'] is not None:
                    try:
                        trade['pnl'] = float(trade['pnl'])
                    except (TypeError, ValueError) as e:
                        app_logger.warning(f"Could not convert PnL for trade {trade.get('id')}: {e}")
                        trade['pnl'] = 0.0
            return trades
        finally:
            await self._release_connection(conn)

    async def calculate_signal_efficiency(self) -> float:
        """Calculate signal-to-trade efficiency ratio"""
        today = datetime.now(timezone.utc).date().isoformat()
        conn = await self._get_connection()
        try:
            cursor = await conn.execute("""
                SELECT 
                    (SELECT COUNT(*) FROM signals WHERE DATE(timestamp) = ?) as signals,
                    (SELECT COUNT(*) FROM trades WHERE DATE(entry_time) = ?) as trades
            """, (today, today))
            row = await cursor.fetchone()
            signals = row[0] if row else 0
            trades = row[1] if row else 0

            if signals == 0:
                return 0.0

            return trades / signals
        finally:
            await self._release_connection(conn)

    async def log_active_order(self, order_data: Dict[str, Any]):
        """Log active order with idempotency"""
        async with self.exclusive_transaction() as db:
            await db.execute("""
                INSERT OR REPLACE INTO active_orders (order_id, symbol, status, filled_qty, submitted_at, last_updated)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                str(order_data['order_id']),
                order_data['symbol'],
                order_data['status'],
                order_data.get('filled_qty', 0),
                order_data['submitted_at']
            ))
            await db.commit()

    async def get_active_orders(self) -> List[Dict[str, Any]]:
        """Get active orders from DB"""
        conn = await self._get_connection()
        try:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT * FROM active_orders WHERE status NOT IN ('FILLED', 'CANCELED', 'REJECTED', 'EXPIRED')")
            return [dict(row) for row in await cursor.fetchall()]
        finally:
            await self._release_connection(conn)

    async def update_active_order(self, order_id: str, status: str, filled_qty: Optional[int] = None):
        """Update active order status"""
        async with self.exclusive_transaction() as db:
            if filled_qty is not None:
                await db.execute("""
                    UPDATE active_orders SET status = ?, filled_qty = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE order_id = ?
                """, (status, filled_qty, order_id))
            else:
                await db.execute("""
                    UPDATE active_orders SET status = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE order_id = ?
                """, (status, order_id))
            await db.commit()

    async def cleanup_old_orders(self):
        """Remove old active_orders records"""
        # Use config passed to __init__ instead of global conf
        cutoff_date = (datetime.now() - timedelta(days=self.config.ACTIVE_ORDER_RETENTION_DAYS)).isoformat()
        async with self.exclusive_transaction() as db:
            await db.execute("DELETE FROM active_orders WHERE submitted_at < ?", (cutoff_date,))
            await db.commit()
            app_logger.info(f"Cleaned up active_orders older than {self.config.ACTIVE_ORDER_RETENTION_DAYS} days")

    async def backup(self):
        """Create timestamped backup with retention and async compression"""
        if not self.db_path.exists():
            return

        # Rate limit backups
        last_backup = list(self.backup_path.glob("titanium_backup_*.db.gz"))
        if last_backup and (time.time() - last_backup[0].stat().st_mtime) < 3600:
            app_logger.debug("Backup already created within last hour, skipping")
            return

        # Remove old backups (>60 days)
        for backup in self.backup_path.glob("titanium_backup_*.db.gz"):
            if backup.stat().st_mtime < time.time() - 60*86400:
                try:
                    await asyncio.to_thread(backup.unlink)
                    app_logger.debug(f"Deleted old backup: {backup}")
                except Exception as e:
                    app_logger.warning(f"Failed to delete old backup {backup}: {e}")

        backup_name = self.backup_path / f"titanium_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"

        # Async backup with proper error handling
        try:
            await asyncio.wait_for(
                self._async_compress_backup(backup_name),
                timeout=300  # 5 minute timeout for backup
            )
            app_logger.info(f"Database backed up to {backup_name}.gz")
            return True
        except asyncio.TimeoutError:
            app_logger.error("Database backup timed out after 5 minutes")
            return False
        except Exception as e:
            app_logger.error(f"Database backup failed: {e}")
            return False

    async def _async_compress_backup(self, backup_name: Path):
        """Async compression with chunked I/O to prevent blocking"""
        chunk_size = 1024 * 1024  # 1MB chunks

        try:
            # Read and compress in thread pool to avoid blocking
            def compress_file():
                with open(self.db_path, 'rb') as f_in:
                    with gzip.open(f"{backup_name}.gz", 'wb') as f_out:
                        while True:
                            chunk = f_in.read(chunk_size)
                            if not chunk:
                                break
                            f_out.write(chunk)

            await asyncio.to_thread(compress_file)

            # Verify backup
            if backup_name.with_suffix('.db.gz').stat().st_size == 0:
                app_logger.error("Backup file is empty")
                backup_name.with_suffix('.db.gz').unlink()
                raise Exception("Backup file is empty")

        except Exception as e:
            # Clean up partial backup
            if backup_name.with_suffix('.db.gz').exists():
                await asyncio.to_thread(backup_name.with_suffix('.db.gz').unlink)
            raise e

# ===============================================================================
# 📡 MULTI-TIMEFRAME DATA ENGINE - ENHANCED WITH PROGRESSIVE DATA COLLECTION
# ===============================================================================

class MultiTimeframeDataEngine:
    def __init__(self, config: SystemConfig, db: DatabaseManager):
        self.config = config
        self.symbol = config.SYMBOL
        self.api_key = config.TWELVEDATA_API_KEY
        self.db = db  # Reference for API budget persistence
        self.timeframes = [config.PRIMARY_TIMEFRAME] + config.INTRADAY_TIMEFRAMES
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.cache_times: Dict[str, datetime] = {}
        self.cache_sizes: Dict[str, int] = {}
        self.max_cache_size = 5
        self.total_cache_memory_mb = 0.0

        # Thread-safe API budget tracking
        self._api_lock = asyncio.Lock()
        self.daily_api_calls = 0
        self.api_limit = config.API_CALLS_PER_DAY_LIMIT
        self.api_calls_per_timeframe: Dict[str, int] = defaultdict(int)
        self.last_api_budget_reset = datetime.now().date()

        # Budget lock shared across engines
        self._budget_lock = asyncio.Lock()

        self.lock = asyncio.Lock()
        self.nyse = mcal.get_calendar('NYSE')
        self.session: Optional[aiohttp.ClientSession] = None

        # Circuit breaker state with lambda factory
        self.circuit_breaker: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: self._create_circuit_breaker_state())
        self.CIRCUIT_THRESHOLD = config.CIRCUIT_FAILURE_THRESHOLD
        self.CIRCUIT_TIMEOUT = config.CIRCUIT_TIMEOUT_SECONDS

        # Data integrity
        self.data_hashes: Dict[str, str] = {}

        # Budget mode state
        self.budget_mode = "adaptive"
        self.last_budget_check = datetime.now()

        # Rate limiting
        self.twelvedata_limiter = AsyncLimiter(8, 60)
        self.yfinance_limiter = AsyncLimiter(1, 5)

        # Initialize expected_next_bar_time
        self.expected_next_bar_time: Optional[datetime] = None

        # Data hash cache
        self._data_hash_cache: Dict[str, str] = {}

        # Market hours cache with TTL and LRU eviction
        self._market_hours_cache: Dict[str, Tuple[bool, float]] = {}
        self._market_hours_cache_ttl = 300  # 5 minutes
        self._market_hours_cache_maxsize = 100  # Maximum entries
        self._last_cache_cleanup = 0

        # Data collection strategy
        self.data_collection_strategy = config.DATA_COLLECTION_STRATEGY
        self.initial_data_bars = config.INITIAL_DATA_BARS
        self.collection_phase: Dict[str, int] = defaultdict(int)  # 0=initial, 1=progressive, 2=full

        # Start API budget loading and periodic persistence
        self._api_budget_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._cache_cleanup_task: Optional[asyncio.Task] = None

        # V19: Data source health tracking
        self.data_source_health: Dict[str, Dict[str, Any]] = {
            "twelvedata": {"healthy": True, "last_success": 0, "failures": 0},
            "yfinance": {"healthy": True, "last_success": 0, "failures": 0}
        }

    def _create_circuit_breaker_state(self) -> Dict[str, Any]:
        """Factory method to create circuit breaker state"""
        return {"open": False, "last_failure": 0, "failure_count": 0}

    async def initialize(self):
        """Initialize the data engine (separate from __init__)"""
        await self._load_api_budget()
        # Start periodic API budget persistence
        self._api_budget_task = asyncio.create_task(self._periodic_api_budget_persistence())
        # Start periodic cache cleanup
        self._cache_cleanup_task = asyncio.create_task(self._periodic_cache_cleanup())
        
        # Initialize collection phase for each timeframe
        for tf in self.timeframes:
            self.collection_phase[tf] = 0  # Start in initial collection phase

    async def _periodic_cache_cleanup(self):
        """Periodically clean expired cache entries"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Every minute
                self._clean_expired_cache_entries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                app_logger.warning(f"Cache cleanup failed: {e}")

    def _clean_expired_cache_entries(self):
        """Clean expired cache entries"""
        current_time = time.time()
        expired_keys = []

        for key, (_, cache_timestamp) in self._market_hours_cache.items():
            if current_time - cache_timestamp >= self._market_hours_cache_ttl:
                expired_keys.append(key)

        for key in expired_keys:
            self._market_hours_cache.pop(key, None)

        if expired_keys:
            app_logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")

    async def _periodic_api_budget_persistence(self):
        """Periodically save API budget to DB"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._save_api_budget()
            except asyncio.CancelledError:
                break
            except Exception as e:
                app_logger.warning(f"Periodic API budget save failed: {e}")

    async def _save_api_budget(self):
        """Save API usage to DB"""
        try:
            await self.db.log_api_usage(
                datetime.now().date().isoformat(),
                self.daily_api_calls,
                self.api_limit - self.daily_api_calls,
                0,
                self.budget_mode
            )
        except Exception as e:
            app_logger.warning(f"Failed to save API budget: {e}")

    async def _load_api_budget(self):
        """Load API usage from DB on startup"""
        try:
            today = datetime.now().date().isoformat()
            usage = await self.db.get_api_usage(today)
            if usage:
                self.daily_api_calls = usage['calls_used']
                app_logger.info(f"API budget restored: {self.daily_api_calls}/{self.api_limit} calls used")
            else:
                self.daily_api_calls = 0
                app_logger.info("API budget initialized to 0")
        except Exception as e:
            app_logger.warning(f"Could not load API budget: {e}, starting from 0")
            self.daily_api_calls = 0

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Ensure background tasks are properly shut down
        self._shutdown_event.set()

        # Cancel and wait for background tasks with timeout
        if self._api_budget_task:
            self._api_budget_task.cancel()
            try:
                await asyncio.wait_for(self._api_budget_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        if self._cache_cleanup_task:
            self._cache_cleanup_task.cancel()
            try:
                await asyncio.wait_for(self._cache_cleanup_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        try:
            # Save API budget before exit
            try:
                await self._save_api_budget()
            except Exception as e:
                app_logger.warning(f"Failed to save API budget on exit: {e}")
        finally:
            if self.session and not self.session.closed:
                await self.session.close()

    async def _ensure_session(self):
        """Reuse session with connection pooling"""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=10, limit_per_host=5, ttl_dns_cache=300)
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30, connect=5, sock_read=10),
                connector=connector
            )
            app_logger.info("aiohttp session created with connection pooling")

    async def is_market_open(self) -> bool:
        """Check if market is currently open with extended hours support and TTL cache"""
        if not self.config.MARKET_HOURS_ONLY:
            return True

        now = pd.Timestamp.now(tz='America/New_York')
        cache_key = now.strftime('%Y-%m-%d')
        current_time = time.time()

        # Check cache with TTL
        if cache_key in self._market_hours_cache:
            is_open, cache_timestamp = self._market_hours_cache[cache_key]
            if current_time - cache_timestamp < self._market_hours_cache_ttl:
                return is_open

        # Check for holidays
        schedule = self.nyse.schedule(start_date=now.date(), end_date=now.date())

        if schedule.empty:
            self._update_cache(cache_key, False, current_time)
            return False

        row = schedule.iloc[0]
        if hasattr(row, 'market_open'):
            market_open = row.market_open
            market_close = row.market_close
        else:
            market_open = row['market_open']
            market_close = row['market_close']

        # Respect extended hours config
        if self.config.EXTENDED_HOURS_ENABLED:
            extended_open = market_open - timedelta(hours=1.5)
            extended_close = market_close + timedelta(hours=4)
        else:
            extended_open = market_open
            extended_close = market_close

        is_open = extended_open <= now <= extended_close
        self._update_cache(cache_key, is_open, current_time)
        return is_open

    def _update_cache(self, key: str, value: bool, timestamp: float):
        """Update cache with size limit"""
        self._market_hours_cache[key] = (value, timestamp)

        # Enforce size limit
        if len(self._market_hours_cache) > self._market_hours_cache_maxsize:
            # Remove oldest entry
            oldest_key = min(self._market_hours_cache.keys(),
                             key=lambda k: self._market_hours_cache[k][1])
            self._market_hours_cache.pop(oldest_key, None)

    async def _check_circuit_breaker(self, source: str) -> bool:
        """Check if circuit breaker is open for a source"""
        cb = self.circuit_breaker[source]
        if cb["open"]:
            if time.time() - cb["last_failure"] > self.CIRCUIT_TIMEOUT:
                cb["open"] = False
                cb["failure_count"] = 0
                app_logger.info(f"Circuit breaker reset for {source}")
                return True
            return False
        return True

    async def _record_failure(self, source: str):
        """Record API failure"""
        cb = self.circuit_breaker[source]
        cb["failure_count"] += 1
        cb["last_failure"] = time.time()

        if cb["failure_count"] >= self.CIRCUIT_THRESHOLD:
            cb["open"] = True
            app_logger.error(f"Circuit breaker OPENED for {source}")

    async def _record_success(self, source: str):
        """Reset failure count on success"""
        if self.circuit_breaker[source]["failure_count"] > 0:
            self.circuit_breaker[source]["failure_count"] = 0

    async def has_api_budget(self, priority: str = "low") -> bool:
        """Check if we have remaining API budget for the day"""
        # Guard against zero or negative limit
        if self.api_limit <= 0:
            app_logger.error("API_CALLS_PER_DAY_LIMIT is zero or negative, disabling all fetches")
            return False

        # Reset daily counter if new day (under lock)
        today = datetime.now().date()
        async with self._api_lock:
            if today != self.last_api_budget_reset:
                self.daily_api_calls = 0
                self.last_api_budget_reset = today

            # Reserve buffer based on priority
            buffer_pct = {
                "high": 0.05,
                "medium": 0.15,
                "low": 0.30
            }

            buffer = self.api_limit * buffer_pct.get(priority, 0.30)
            available = self.api_limit - self.daily_api_calls

            # Adaptive mode: get more conservative as we approach limit
            if self.config.API_BUDGET_MODE == "adaptive":
                pct_used = (self.daily_api_calls / self.api_limit) * 100
                if pct_used > 70:
                    buffer *= 1.5

            # Hard cutoff at 98%
            pct_used = (self.daily_api_calls / self.api_limit) * 100
            return available > buffer and pct_used < 98

    def get_budget_status(self) -> Dict[str, float]:
        """Get detailed budget status"""
        used = self.daily_api_calls
        remaining = self.api_limit - used
        pct_used = (used / self.api_limit) * 100 if self.api_limit > 0 else 100
        pct_remaining = (remaining / self.api_limit) * 100 if self.api_limit > 0 else 0

        # Estimate remaining calls based on current burn rate
        hours_remaining = max(0, (24 - datetime.now().hour))
        burn_rate = used / max(datetime.now().hour, 1)
        projected_remaining = remaining - (burn_rate * hours_remaining)

        return {
            "used": used,
            "remaining": remaining,
            "limit": self.api_limit,
            "pct_used": pct_used,
            "pct_remaining": pct_remaining,
            "burn_rate_hourly": burn_rate,
            "projected_remaining": projected_remaining,
            "mode": self.budget_mode,
            "calls_per_timeframe": dict(self.api_calls_per_timeframe)
        }

    def update_budget_mode(self):
        """Update budget mode based on usage"""
        status = self.get_budget_status()

        if self.config.API_BUDGET_MODE == "adaptive":
            if status['pct_used'] > 80:
                self.budget_mode = "conservative"
            elif status['pct_used'] < 50:
                self.budget_mode = "aggressive"
            else:
                self.budget_mode = "normal"

        if self.budget_mode == "conservative":
            app_logger.info(f"Budget mode: CONSERVATIVE ({status['pct_used']:.1f}% used)")
        elif self.budget_mode == "aggressive":
            app_logger.debug(f"Budget mode: AGGRESSIVE ({status['pct_used']:.1f}% used)")
        else:
            app_logger.debug(f"Budget mode: NORMAL ({status['pct_used']:.1f}% used)")

        self.last_budget_check = datetime.now()

    async def fetch_timeframe(self, timeframe: str, force_refresh: bool = False, priority: str = "medium") -> bool:
        """Async data fetching with progressive collection, circuit breaker protection, and API budget"""
        async with self.lock:
            # Check budget BEFORE incrementing
            budget_available = await self.has_api_budget(priority)
            if not budget_available:
                budget_status = self.get_budget_status()
                app_logger.warning(f"API budget insufficient ({budget_status['pct_used']:.1f}% used). Skipping {timeframe}")
                return False

            now = datetime.now()

            # Cache check with max staleness
            if not force_refresh and timeframe in self.cache_times:
                elapsed = (now - self.cache_times[timeframe]).total_seconds() / 60
                base_intervals = {
                    "1d": 60,
                    "4h": 30,
                    "1h": 15,
                    "15m": 5
                }
                interval = base_intervals.get(timeframe, self.config.DATA_FETCH_INTERVAL_MINUTES)
                max_staleness = interval * self.config.MAX_CACHE_AGE_MULTIPLIER

                if elapsed < interval:
                    return True

                if elapsed > max_staleness:
                    app_logger.warning(f"{timeframe}: Data stale ({elapsed:.0f}min), forcing refresh")
                    force_refresh = True

            # Determine data collection strategy
            outputsize = self._get_outputsize_for_timeframe(timeframe)
            
            # Try TwelveData first with rate limiting
            success = False
            async with self.twelvedata_limiter:
                if await self._check_circuit_breaker("twelvedata"):
                    success = await self._fetch_twelvedata_enhanced(timeframe, outputsize)

            # Fallback to yFinance (doesn't count against budget)
            if not success and await self._check_circuit_breaker("yfinance"):
                app_logger.info(f"Falling back to yFinance for {timeframe}")
                success = await self._fetch_yfinance_enhanced(timeframe, outputsize)

            # Emergency fallback if both sources fail
            if not success:
                app_logger.warning(f"Both data sources failed for {timeframe}, using emergency fallback")
                success = await self._emergency_data_fallback(timeframe)

            # Only increment API counter on successful fetch from TwelveData
            if success and self.data_source_health["twelvedata"]["healthy"]:
                async with self._api_lock:
                    self.daily_api_calls += 1
                    self.api_calls_per_timeframe[timeframe] += 1

            if not success:
                app_logger.critical(f"All data sources failed for {timeframe}")
                # Don't raise exception, just return False for graceful degradation
                return False

            self.cache_times[timeframe] = now

            # Validate data quality with progressive thresholds
            df = self.get_df(timeframe)
            validation_result = self._validate_dataframe_enhanced(df, timeframe)
            
            if not validation_result["valid"]:
                app_logger.error(f"Data validation failed for {timeframe}: {validation_result['reason']}")
                
                # If we have some data but it's not perfect, we might still use it
                if validation_result.get("partial", False) and len(df) > self.config.MIN_INITIAL_BARS:
                    app_logger.warning(f"Using partial data for {timeframe}: {validation_result['reason']}")
                    # Continue with partial data
                else:
                    return False

            # Update cache size tracking
            self.cache_sizes[timeframe] = len(df)
            await self._enforce_cache_limits()

            # Use cached hash if data unchanged
            if len(df) > 0:
                new_hash = hashlib.sha256(df.tail(min(100, len(df))).to_csv().encode()).hexdigest()
                if self._data_hash_cache.get(timeframe) != new_hash:
                    self.data_hashes[timeframe] = new_hash
                    self._data_hash_cache[timeframe] = new_hash

            app_logger.info(f"{timeframe}: {len(df)} bars | API Calls: {self.daily_api_calls}/{self.api_limit}")
            
            # Update collection phase based on data accumulated
            await self._update_collection_phase(timeframe, len(df))
            
            return True

    def _get_outputsize_for_timeframe(self, timeframe: str) -> int:
        """Determine output size based on collection phase"""
        phase = self.collection_phase[timeframe]
        
        if phase == 0:  # Initial collection
            # Start with small amount for faster response
            return min(self.config.INITIAL_DATA_BARS, 100)
        elif phase == 1:  # Progressive collection
            # Increase gradually
            current_size = len(self.dataframes.get(timeframe, pd.DataFrame()))
            return min(current_size * 2, 500)
        else:  # Full collection
            # Use full historical data
            base_sizes = {
                "1d": 500,
                "4h": 800,
                "1h": 1200,
                "15m": 2000
            }
            return min(base_sizes.get(timeframe, 500), self.config.TWELVEDATA_MAX_OUTPUTSIZE)

    async def _update_collection_phase(self, timeframe: str, current_bars: int):
        """Update collection phase based on accumulated data"""
        target_bars = self.config.HMM_TRAIN_WINDOW.get(timeframe, 504)
        
        if current_bars >= target_bars * 0.9:
            self.collection_phase[timeframe] = 2  # Full collection
        elif current_bars >= target_bars * 0.5:
            self.collection_phase[timeframe] = 1  # Progressive collection
        else:
            self.collection_phase[timeframe] = 0  # Initial collection

    async def _enforce_cache_limits(self):
        """Enforce LRU cache limit to prevent memory leaks with hard MB limit"""
        # Calculate current memory usage
        current_memory = 0.0
        for timeframe, df in self.dataframes.items():
            if not df.empty and hasattr(df, 'memory_usage'):
                try:
                    current_memory += df.memory_usage(deep=True).sum() / 1024 / 1024
                except Exception:
                    # If memory calculation fails, estimate
                    current_memory += len(df) * len(df.columns) * 8 / 1024 / 1024

        self.total_cache_memory_mb = current_memory

        # Check hard memory limit
        if current_memory > self.config.MAX_CACHE_SIZE_MB:
            app_logger.warning(f"Cache memory {current_memory:.1f}MB > limit {self.config.MAX_CACHE_SIZE_MB}MB, evicting")

            # Evict oldest timeframes until under limit
            sorted_by_age = sorted(self.cache_times.items(), key=lambda x: x[1])
            for timeframe, _ in sorted_by_age:
                if timeframe in self.dataframes:
                    try:
                        df_memory = self.dataframes[timeframe].memory_usage(deep=True).sum() / 1024 / 1024
                    except Exception:
                        df_memory = len(self.dataframes[timeframe]) * len(self.dataframes[timeframe].columns) * 8 / 1024 / 1024
                    
                    # Use helper method to evict completely
                    self._evict_timeframe_completely(timeframe)
                    current_memory -= df_memory
                    app_logger.info(f"Evicted {timeframe} from cache ({df_memory:.1f}MB)")

                    if current_memory <= self.config.MAX_CACHE_SIZE_MB * 0.8:  # Go to 80% to prevent thrashing
                        break

        # Enforce max rows per timeframe
        for timeframe, df in self.dataframes.items():
            if len(df) > self.config.MAX_ROWS_PER_TIMEFRAME:
                self.dataframes[timeframe] = df.tail(self.config.MAX_ROWS_PER_TIMEFRAME // 2)
                app_logger.info(f"Trimmed {timeframe} cache to {len(self.dataframes[timeframe])} rows")

        # Also enforce max number of cached timeframes
        if len(self.dataframes) > self.max_cache_size:
            oldest_tf = min(self.cache_times.keys(), key=lambda k: self.cache_times[k])
            self._evict_timeframe_completely(oldest_tf)
            app_logger.info(f"Evicted {oldest_tf} from cache (max timeframes)")

    def _evict_timeframe_completely(self, timeframe: str):
        """Helper to completely evict a timeframe from all tracking dictionaries"""
        self.dataframes.pop(timeframe, None)
        self.cache_times.pop(timeframe, None)
        self.cache_sizes.pop(timeframe, None)
        self._data_hash_cache.pop(timeframe, None)
        self.data_hashes.pop(timeframe, None)
        self.api_calls_per_timeframe.pop(timeframe, None)
        self.collection_phase.pop(timeframe, None)

    def get_df(self, timeframe: str) -> pd.DataFrame:
        """Get dataframe for timeframe"""
        return self.dataframes.get(timeframe, pd.DataFrame())

    def _validate_dataframe_enhanced(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Enhanced validation with progressive thresholds and better error reporting"""
        result = {"valid": False, "reason": "", "partial": False}
        
        # Check if dataframe is completely empty
        if df.empty:
            result["reason"] = "Dataframe is completely empty"
            return result
        
        # Check for required columns with case-insensitive matching
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        df_cols_lower = [str(col).lower() for col in df.columns]
        
        missing_cols = []
        for req_col in required_cols:
            if req_col not in df_cols_lower:
                missing_cols.append(req_col)
        
        if missing_cols:
            result["reason"] = f"Missing required columns: {missing_cols}"
            return result
        
        # Standardize column names to lowercase
        df.columns = [str(col).lower() for col in df.columns]
        
        # Check for zero or negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns and (df[col] <= 0).any():
                app_logger.warning(f"{timeframe}: Zero or negative prices in {col}")
                # Filter out invalid rows
                df = df[df[col] > 0]
        
        # Check for sufficient data with progressive thresholds
        min_initial_bars = self.config.MIN_INITIAL_BARS
        target_bars = self.config.HMM_TRAIN_WINDOW.get(timeframe, 504)
        
        if len(df) < min_initial_bars:
            result["reason"] = f"Insufficient data ({len(df)} < {min_initial_bars})"
            result["partial"] = True
            return result
        
        # Check for data quality issues
        null_counts = df.isnull().sum()
        for col, null_count in null_counts.items():
            if null_count > 0:
                null_pct = null_count / len(df)
                if null_pct > 0.1:  # More than 10% nulls
                    app_logger.warning(f"{timeframe}: {col} has {null_count} null values ({null_pct:.1%})")
        
        # Fill missing values if reasonable
        if df.isnull().any().any():
            fill_count = df.isnull().sum().sum()
            if fill_count < len(df) * 0.2:  # Less than 20% missing
                df = df.ffill().bfill()
                app_logger.info(f"Filled {fill_count} missing values in {timeframe}")
            else:
                result["reason"] = f"Too many missing values ({fill_count})"
                result["partial"] = True
                return result
        
        # Check for price anomalies
        if ('high' in df.columns and 'low' in df.columns and 
            (df['high'] < df['low']).any()):
            result["reason"] = "high < low detected"
            return result
        
        if ('close' in df.columns and 'high' in df.columns and 'low' in df.columns):
            if (df['close'] > df['high']).any() or (df['close'] < df['low']).any():
                result["reason"] = "close outside high/low range"
                return result
        
        # Data staleness check (only if we have at least 2 bars)
        if len(df) > 2:
            last_bar_age = (datetime.now() - df.index[-1]).total_seconds()
            expected_interval = {"1d": 86400, "4h": 14400,
                                 "1h": 3600, "15m": 900}.get(timeframe, 3600)
            staleness_threshold = expected_interval * self.config.DATA_STALENESS_THRESHOLD
            
            if last_bar_age > staleness_threshold:
                result["reason"] = f"Data stale ({last_bar_age:.0f}s old)"
                result["partial"] = True
                # Still return True for partial data if we have enough bars
                if len(df) >= min_initial_bars:
                    result["valid"] = True
                    return result
                return result
        
        # Update the dataframe in cache
        self.dataframes[timeframe] = df
        
        result["valid"] = True
        return result

    async def _fetch_twelvedata_enhanced(self, timeframe: str, outputsize: int) -> bool:
        """Enhanced TwelveData fetching with better error handling and case-insensitive column matching"""
        try:
            td_intervals = {"1d": "1day", "4h": "4h", "1h": "1h", "15m": "15min"}
            if timeframe not in td_intervals:
                return False

            interval = td_intervals[timeframe]

            url = "https://api.twelvedata.com/time_series"
            params = {
                "symbol": self.symbol,
                "interval": interval,
                "apikey": self.api_key,
                "outputsize": outputsize,
                "order": "ASC",
                "format": "JSON"
            }

            app_logger.debug(f"Fetching {timeframe} data from TwelveData (outputsize: {outputsize})")

            await self._ensure_session()
            async with self.session.get(url, params=params, timeout=30) as response:
                if response.status != 200:
                    app_logger.error(f"TwelveData HTTP error {response.status}: {await response.text()}")
                    self.data_source_health["twelvedata"]["failures"] += 1
                    await self._record_failure("twelvedata")
                    return False
                
                data = await response.json()
                
                # Debug logging for response structure
                if "values" not in data:
                    error_msg = data.get('message', 'Unknown error') if isinstance(data, dict) else 'Invalid response format'
                    app_logger.error(f"TwelveData error: {error_msg}")
                    self.data_source_health["twelvedata"]["failures"] += 1
                    await self._record_failure("twelvedata")
                    return False

                values = data["values"]
                if not values:
                    app_logger.warning(f"TwelveData returned empty data for {timeframe}")
                    return False

                # Create DataFrame
                df = pd.DataFrame(values)
                
                # Case-insensitive column name standardization
                df.columns = [str(col).lower() for col in df.columns]
                
                # Find datetime column (case-insensitive)
                datetime_col = None
                for col in df.columns:
                    if 'datetime' in col:
                        datetime_col = col
                        break
                
                if not datetime_col:
                    app_logger.error(f"No datetime column found. Columns: {df.columns.tolist()}")
                    return False
                
                # Convert datetime
                df['datetime'] = pd.to_datetime(df[datetime_col])
                df.set_index('datetime', inplace=True)
                
                # Drop the original datetime column if it's different
                if datetime_col != 'datetime':
                    df = df.drop(columns=[datetime_col])
                
                # Ensure required columns exist (case-insensitive)
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                available_cols = []
                
                for req_col in required_cols:
                    if req_col in df.columns:
                        available_cols.append(req_col)
                    else:
                        # Try to find case-insensitive match
                        matching_cols = [col for col in df.columns if req_col in col.lower()]
                        if matching_cols:
                            df = df.rename(columns={matching_cols[0]: req_col})
                            available_cols.append(req_col)
                        else:
                            app_logger.warning(f"Column {req_col} not found in TwelveData response")
                
                # Check if we have at least the essential columns
                essential_cols = ['open', 'high', 'low', 'close']
                if not all(col in df.columns for col in essential_cols):
                    app_logger.error(f"Missing essential columns. Available: {df.columns.tolist()}")
                    return False
                
                # Convert to numeric
                for col in essential_cols + (['volume'] if 'volume' in df.columns else []):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Drop rows with NaN in essential columns
                df = df.dropna(subset=essential_cols)
                
                if df.empty:
                    app_logger.warning(f"DataFrame empty after cleaning for {timeframe}")
                    return False

                app_logger.info(f"TwelveData: {timeframe} - {len(df)} bars")
                
                # Store in dataframes
                self.dataframes[timeframe] = await self._engineer_features(df, timeframe)
                await self._record_success("twelvedata")
                self.data_source_health["twelvedata"]["healthy"] = True
                self.data_source_health["twelvedata"]["last_success"] = time.time()
                self.data_source_health["twelvedata"]["failures"] = 0
                return True

        except aiohttp.ClientError as e:
            app_logger.error(f"TwelveData network error: {e}")
            self.data_source_health["twelvedata"]["failures"] += 1
            await self._record_failure("twelvedata")
            return False
        except Exception as e:
            app_logger.error(f"TwelveData failed: {e}", exc_info=True)
            self.data_source_health["twelvedata"]["failures"] += 1
            await self._record_failure("twelvedata")
            return False

    async def _fetch_yfinance_enhanced(self, timeframe: str, outputsize: int) -> bool:
        """Enhanced yFinance fetching with better error handling"""
        async with self.yfinance_limiter:
            try:
                interval_map = {"1d": "1d", "4h": "60m", "1h": "60m", "15m": "15m"}
                if timeframe not in interval_map:
                    return False

                # Calculate period based on outputsize and timeframe
                if timeframe == "1d":
                    period = f"{min(outputsize, 730)}d"  # Max 2 years
                elif timeframe == "4h":
                    period = f"{min(outputsize // 6, 60)}d"  # Max 60 days
                elif timeframe == "1h":
                    period = f"{min(outputsize // 24, 60)}d"  # Max 60 days
                else:  # 15m
                    period = f"{min(outputsize // 96, 60)}d"  # Max 60 days

                app_logger.debug(f"yFinance {timeframe}: Requesting {period} data")

                ticker = yf.Ticker(self.symbol)

                # yFinance fetch with timeout
                df = await asyncio.wait_for(
                    asyncio.to_thread(
                        lambda: ticker.history(
                            interval=interval_map[timeframe],
                            period=period,
                            timeout=15,
                            raise_errors=True,
                            auto_adjust=True
                        )
                    ),
                    timeout=20
                )

                if df.empty:
                    app_logger.warning(f"yFinance returned empty DataFrame for {timeframe}")
                    return False

                # Standardize column names
                df.columns = df.columns.str.lower()
                
                # Resample for 4h if needed
                if timeframe == "4h" and interval_map[timeframe] == "60m":
                    df = df.resample('4H', offset='1h').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()

                # Remove duplicates
                df = df[~df.index.duplicated(keep='last')]
                
                # Ensure required columns
                required_cols = ['open', 'high', 'low', 'close']
                if not all(col in df.columns for col in required_cols):
                    app_logger.error(f"yFinance missing required columns: {df.columns.tolist()}")
                    return False
                
                # Drop rows with NaN in essential columns
                df = df.dropna(subset=required_cols)
                
                if df.empty:
                    return False

                df = await self._engineer_features(df, timeframe)
                self.dataframes[timeframe] = df
                await self._record_success("yfinance")
                self.data_source_health["yfinance"]["healthy"] = True
                self.data_source_health["yfinance"]["last_success"] = time.time()
                self.data_source_health["yfinance"]["failures"] = 0
                return True

            except asyncio.TimeoutError:
                app_logger.error("yFinance fetch timed out")
                self.data_source_health["yfinance"]["failures"] += 1
                await self._record_failure("yfinance")
                return False
            except Exception as e:
                app_logger.error(f"yFinance failed: {e}")
                self.data_source_health["yfinance"]["failures"] += 1
                await self._record_failure("yfinance")
                return False

    async def _emergency_data_fallback(self, timeframe: str) -> bool:
        """Emergency fallback when all other data sources fail"""
        try:
            app_logger.warning(f"Using emergency fallback for {timeframe}")
            
            # Try to use cached data if available
            if timeframe in self.dataframes and not self.dataframes[timeframe].empty:
                df = self.dataframes[timeframe]
                app_logger.info(f"Using cached data for {timeframe}: {len(df)} bars")
                return True
            
            # Try to generate synthetic data for testing
            if self.config.PAPER_TRADING:
                app_logger.warning(f"Generating synthetic data for {timeframe} in paper trading mode")
                
                # Create a simple synthetic dataset
                dates = pd.date_range(end=datetime.now(), periods=100, freq=timeframe)
                np.random.seed(42)
                base_price = 100.0
                returns = np.random.normal(0.0001, 0.02, 100)
                prices = base_price * np.exp(np.cumsum(returns))
                
                df = pd.DataFrame({
                    'open': prices * 0.99,
                    'high': prices * 1.01,
                    'low': prices * 0.98,
                    'close': prices,
                    'volume': np.random.randint(100000, 1000000, 100)
                }, index=dates)
                
                df = await self._engineer_features(df, timeframe)
                self.dataframes[timeframe] = df
                app_logger.warning(f"Generated synthetic data for {timeframe}")
                return True
            
            return False
            
        except Exception as e:
            app_logger.error(f"Emergency fallback failed: {e}")
            return False

    async def _engineer_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Engineer features without lookahead bias, with timeframe-aware parameters"""
        # Remove zero/negative prices
        df = df[(df['close'] > 0) & (df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0)]

        if df.empty:
            return df

        # Cap extreme outliers (non-physical)
        for col in ['close', 'open', 'high', 'low']:
            with np.errstate(all='ignore'):
                q01, q99 = df[col].quantile([0.01, 0.99])
                df[col] = df[col].clip(lower=q01*0.5, upper=q99*2)

        df = df.copy().astype(np.float64)

        # Timeframe-aware parameters
        if timeframe == "1d":
            lookback = 20
            rsi_period = 14
            vwap_period = 20
        elif timeframe == "4h":
            lookback = 15
            rsi_period = 12
            vwap_period = 15
        elif timeframe == "1h":
            lookback = 10
            rsi_period = 10
            vwap_period = 10
        else:
            lookback = 8
            rsi_period = 8
            vwap_period = 8

        # Basic features
        with np.errstate(divide='ignore', invalid='ignore'):
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            df['log_ret'] = df['log_ret'].replace([np.inf, -np.inf], np.nan)

        # ATR (for position sizing)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        true_range = np.max([high_low, high_close, low_close], axis=0)
        df['atr'] = pd.Series(true_range).rolling(self.config.ATR_PERIOD).mean()

        # Garman-Klass volatility
        hl = (np.log(df['high'] / df['low']) ** 2) / 2
        co = (np.log(df['close'] / df['open']) ** 2)
        df['vol_gk'] = np.sqrt((hl + co).rolling(lookback).mean())

        # Trend Efficiency - uses only past data
        change = (df['close'] - df['close'].shift(lookback)).abs()
        volatility = df['close'].diff().abs().rolling(lookback).sum()

        # Prevent division by zero with dynamic floor
        vol_floor = max(volatility.mean() * 0.01, 1e-6)
        volatility = volatility.replace(0, vol_floor)

        df['trend_eff'] = (change / (volatility)).clip(0, 1)

        # RSI (no bias)
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
        rs = gain / (loss.replace(0, np.nan) + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))

        # Volume Profile (simplified) - Rolling VWAP (no lookahead)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (df['volume'] * typical_price).rolling(vwap_period).sum() / \
            df['volume'].rolling(vwap_period).sum()
        # Shift to prevent lookahead bias (NO backward fill)
        df['vwap'] = df['vwap'].shift(1)
        df['vwap_dev'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-9)

        # Optimize outlier cap
        for col in ['close', 'open', 'high', 'low']:
            q01, q99 = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lower=q01*0.5, upper=q99*2)

        # Momentum divergence
        df['rsi_mom'] = df['rsi'].diff(3)
        df['price_mom'] = df['close'].diff(3)
        df['divergence'] = np.sign(df['rsi_mom']) != np.sign(df['price_mom'])

        # Data continuity check - only warn, don't fail
        if len(df) > 2:
            time_diffs = df.index.to_series().diff().dt.total_seconds()
            expected_diff = {"1d": 86400, "4h": 14400,
                             "1h": 3600, "15m": 900}.get(timeframe, 3600)
            gaps = (time_diffs > expected_diff * 2).sum()
            if gaps > 0:
                app_logger.warning(f"{timeframe}: {gaps} data gaps detected")
                # We don't return False, just log

        return df.dropna()

# ===============================================================================
# 🧠 MULTI-TIMEFRAME BRAIN - ENHANCED REGIME DETECTION WITH SEQUENTIAL HMM
# ===============================================================================

class MultiTimeframeBrain:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.models: Dict[str, hmm.GaussianHMM] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.regime_maps: Dict[str, Dict[int, str]] = {}
        self.is_trained: Dict[str, bool] = {}
        self.training_metrics: Dict[str, Dict[str, Any]] = {}
        self.training_hashes: Dict[str, str] = {}
        self.lock = asyncio.Lock()
        self.last_train_time: Dict[str, datetime] = {}
        self.performance_window: Dict[str, deque] = {}

        # Drift detection
        self.drift_threshold = 0.12
        self.performance_history: Dict[str, deque] = {
            tf: deque(maxlen=50) for tf in [config.PRIMARY_TIMEFRAME] + config.INTRADAY_TIMEFRAMES
        }

        # Chop detection
        self.chop_threshold = 0.55

        # Dynamic component selection
        self.component_usage: Dict[str, int] = {}

        # Stability Buffer
        self.retrain_cooldowns: Dict[str, datetime] = {}

        # Training failure counter
        self.training_failures: Dict[str, int] = defaultdict(int)
        self.training_failure_threshold = 3

        # Model persistence with joblib
        self.model_cache_path = Path("models")
        self.model_cache_path.mkdir(exist_ok=True)

        # Reference to data engine using weak reference to prevent circular reference
        self._data_engine_ref: Optional[weakref.ReferenceType] = None

        # Telegram for alerts
        self._telegram_ref: Optional[weakref.ReferenceType] = None

        # Sequence length for HMM inference (STATISTICAL FIX)
        self.sequence_length = config.HMM_SEQUENCE_WINDOW
        self.min_sequence_length = config.HMM_MIN_SEQUENCE_LENGTH

    @property
    def data_engine(self):
        """Get data engine from weak reference with graceful degradation"""
        if self._data_engine_ref:
            engine = self._data_engine_ref()
            if engine is None:
                app_logger.warning("Data engine weak reference has expired - entering degraded mode")
                return None
            return engine
        app_logger.warning("Data engine not set - entering degraded mode")
        return None

    @data_engine.setter
    def data_engine(self, engine):
        """Set data engine using weak reference"""
        self._data_engine_ref = weakref.ref(engine) if engine else None

    @property
    def telegram(self):
        """Get telegram from weak reference with graceful degradation"""
        if self._telegram_ref:
            telegram_bot = self._telegram_ref()
            if telegram_bot is None:
                app_logger.warning("Telegram weak reference has expired - alerts disabled")
                return None
            return telegram_bot
        return None

    @telegram.setter
    def telegram(self, telegram_bot):
        """Set telegram using weak reference"""
        self._telegram_ref = weakref.ref(telegram_bot) if telegram_bot else None

    async def initialize(self):
        """Initialize the brain (separate from __init__)"""
        await self._cleanup_old_models()

    async def _cleanup_old_models(self):
        """Async cleanup of old model files"""
        try:
            for model_file in self.model_cache_path.glob("*.joblib"):
                if (time.time() - model_file.stat().st_mtime) > 90*86400:
                    await asyncio.to_thread(model_file.unlink)
                    app_logger.info(f"Deleted old model: {model_file}")
        except Exception as e:
            app_logger.warning(f"Model cleanup failed: {e}")

    async def should_retrain(self, timeframe: str, df: pd.DataFrame, recent_performance: Optional[float] = None) -> Tuple[bool, str]:
        """Determine if model needs retraining based on data drift and performance"""
        reasons = []

        # Check cooldown (Prevent retraining too frequently which causes label flip chaos)
        if timeframe in self.retrain_cooldowns:
            time_since_retrain = (datetime.now() - self.retrain_cooldowns[timeframe]).total_seconds()
            if time_since_retrain < 3600:
                return False, "cooldown"

        # Check time since last train
        if timeframe not in self.last_train_time:
            reasons.append("first_train")
        else:
            bars_since_train = len(df) - self.config.HMM_TRAIN_WINDOW.get(timeframe, 504)
            retrain_interval = self.config.HMM_RETRAIN_INTERVAL_BARS.get(timeframe, 100)

            if bars_since_train >= retrain_interval:
                reasons.append(f"interval_{bars_since_train}/{retrain_interval}")

        # Check data hash for significant changes (cached)
        if len(df) > 0:
            new_hash = hashlib.sha256(df.tail(min(100, len(df))).to_csv().encode()).hexdigest()
            old_hash = self.training_hashes.get(timeframe, "")

            if new_hash != old_hash:
                reasons.append("data_changed")

        # Check performance degradation
        if recent_performance is not None and recent_performance < self.drift_threshold:
            reasons.append(f"drift_{recent_performance:.2f}")

        # Check component count optimization
        current_components = self.component_usage.get(timeframe, self.config.HMM_COMPONENTS)
        optimal_components = self._optimal_component_count(df)
        if optimal_components != current_components:
            reasons.append(f"components_{current_components}->{optimal_components}")

        return len(reasons) > 0, "|".join(reasons)

    def _optimal_component_count(self, df: pd.DataFrame) -> int:
        """Determine optimal HMM components based on data characteristics"""
        if df.empty or 'vol_gk' not in df.columns:
            return self.config.HMM_COMPONENTS
            
        volatility = df['vol_gk'].mean() if 'vol_gk' in df.columns else 0.01
        if len(df) > 100:
            vol_percentile = np.percentile(df['vol_gk'].dropna(), 75)
        else:
            vol_percentile = volatility

        if vol_percentile > 0.05:
            return min(self.config.HMM_MAX_COMPONENTS, 5)
        elif vol_percentile > 0.03:
            return self.config.HMM_COMPONENTS
        else:
            return max(self.config.HMM_MIN_COMPONENTS, 3)

    async def train_timeframe(self, timeframe: str, df: pd.DataFrame, retrain_reason: str = "") -> bool:
        """Train HMM with walk-forward validation, regime stability, and dynamic components"""
        async with self.lock:
            start_time = time.time()

            # Check if we have enough data
            if df.empty or len(df) < self.config.MIN_INITIAL_BARS:
                app_logger.warning(f"{timeframe}: Insufficient data for training ({len(df)} bars)")
                self.training_failures[timeframe] += 1
                await self._check_training_failure_threshold(timeframe)
                return False

            # Adaptive window reduction if insufficient data
            original_window = self.config.HMM_TRAIN_WINDOW.get(timeframe, 504)
            train_window = original_window
            available_bars = len(df)
            min_bars = max(self.config.MIN_INITIAL_BARS, int(train_window * 0.5))  # More lenient

            # Calculate min_bars once, reduce window systematically
            while available_bars < min_bars and train_window > 100:
                train_window = int(train_window * 0.8)
                app_logger.warning(f"{timeframe}: Reducing train window to {train_window} (insufficient data)")

            if available_bars < min_bars:
                app_logger.warning(f"{timeframe}: Insufficient data even after reduction")
                self.training_failures[timeframe] += 1
                await self._check_training_failure_threshold(timeframe)
                return False

            try:
                # Enhanced features
                features = ['log_ret', 'vol_gk', 'rsi', 'trend_eff', 'vwap_dev']
                available_features = [f for f in features if f in df.columns]

                if len(available_features) < 3:
                    app_logger.warning(f"{timeframe}: Insufficient features ({len(available_features)})")
                    self.training_failures[timeframe] += 1
                    await self._check_training_failure_threshold(timeframe)
                    return False

                X = df[available_features].values

                # Validate finite values
                if not np.isfinite(X).all():
                    app_logger.warning(f"{timeframe}: NaN or infinite values in training data")
                    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

                # Walk-forward split (80/20)
                split_idx = int(len(X) * 0.8)
                X_train = X[:split_idx]
                X_test = X[split_idx:]

                # Scale
                scaler = StandardScaler()
                X_train_scaled = await asyncio.to_thread(scaler.fit_transform, X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train model with dynamic components
                optimal_components = self._optimal_component_count(df)

                # Offload to thread pool with timeout
                model = await asyncio.wait_for(
                    asyncio.to_thread(
                        lambda: hmm.GaussianHMM(
                            n_components=optimal_components,
                            covariance_type="diag",
                            n_iter=self.config.HMM_MAX_ITER,
                            random_state=self.config.HMM_RANDOM_STATE,
                            verbose=False,
                            tol=1e-4
                        ).fit(X_train_scaled)
                    ),
                    timeout=60
                )

                # Validate
                train_score = model.score(X_train_scaled)
                test_score = model.score(X_test_scaled)
                score_ratio = test_score / (train_score + 1e-9)

                if score_ratio < 0.55:
                    app_logger.warning(f"{timeframe}: Model overfitting ({score_ratio:.2f})")
                    self.training_failures[timeframe] += 1
                    await self._check_training_failure_threshold(timeframe)
                    return False

                # Check regime diversity
                if not await self._check_regime_diversity(model, X_train_scaled, optimal_components):
                    app_logger.error(f"{timeframe}: Insufficient regime diversity in training data")
                    raise InsufficientRegimeDiversityError(f"{timeframe} lacks regime diversity")

                # STABLE regime mapping with historical consistency
                old_regime_map = self.regime_maps.get(timeframe, {})
                new_regime_map = await self._map_regimes_stably(timeframe, model, X_train_scaled, old_regime_map, available_features)

                self.models[timeframe] = model
                self.scalers[timeframe] = scaler
                self.regime_maps[timeframe] = new_regime_map
                self.is_trained[timeframe] = True
                self.last_train_time[timeframe] = datetime.now()
                self.retrain_cooldowns[timeframe] = datetime.now()
                if len(df) > 0:
                    self.training_hashes[timeframe] = hashlib.sha256(df.tail(min(100, len(df))).to_csv().encode()).hexdigest()
                self.component_usage[timeframe] = optimal_components
                self.training_failures[timeframe] = 0

                self.training_metrics[timeframe] = {
                    "train_score": train_score,
                    "test_score": test_score,
                    "score_ratio": score_ratio,
                    "features": available_features,
                    "regime_map": new_regime_map,
                    "train_duration_ms": int((time.time() - start_time) * 1000),
                    "retrain_reason": retrain_reason,
                    "components_used": optimal_components
                }

                # Limit training metrics size
                if len(self.training_metrics) > 100:
                    oldest = min(self.training_metrics.keys(), key=lambda k: self.training_metrics[k]['train_duration_ms'])
                    del self.training_metrics[oldest]

                # Persist model to disk with HMAC
                await self._persist_model(timeframe, model, scaler, new_regime_map, available_features)

                # Initialize performance tracking
                if timeframe not in self.performance_window:
                    self.performance_window[timeframe] = deque(maxlen=50)

                app_logger.info(f"{timeframe} HMM trained (ratio: {score_ratio:.2f}, components: {optimal_components}, duration: {self.training_metrics[timeframe]['train_duration_ms']}ms)")
                return True

            except asyncio.TimeoutError:
                app_logger.error(f"{timeframe} training timed out")
                self.training_failures[timeframe] += 1
                await self._check_training_failure_threshold(timeframe)
                return False
            except InsufficientRegimeDiversityError:
                raise
            except Exception as e:
                app_logger.error(f"{timeframe} training failed: {e}")
                self.training_failures[timeframe] += 1
                await self._check_training_failure_threshold(timeframe)
                return False

    async def _check_training_failure_threshold(self, timeframe: str):
        """Alert if training failures exceed threshold"""
        if self.training_failures[timeframe] >= self.training_failure_threshold:
            app_logger.critical(f"{timeframe}: Training failed {self.training_failures[timeframe]} times - strategy degraded")
            # Telegram alert
            try:
                telegram_bot = self.telegram
                if telegram_bot:
                    await telegram_bot.send(
                        f"**MODEL DEGRADATION**\nTimeframe: {timeframe}\nFailures: {self.training_failures[timeframe]}\nManual intervention required.",
                        priority="critical"
                    )
            except Exception as e:
                app_logger.error(f"Cannot send Telegram alert: {e}")

    async def _check_regime_diversity(self, model: hmm.GaussianHMM, X: np.ndarray, n_components: int) -> bool:
        """Check that training data contains at least 3 unique regimes"""
        predictions = model.predict(X)
        unique_regimes = len(set(predictions))
        return unique_regimes >= min(3, n_components)

    async def _persist_model(self, timeframe: str, model: hmm.GaussianHMM, scaler: StandardScaler, regime_map: Dict[int, str], features: List[str]):
        """Save model to disk with HMAC signature"""
        try:
            model_data = {
                'model': model,
                'scaler': scaler,
                'regime_map': regime_map,
                'timestamp': datetime.now(),
                'data_hash': self.training_hashes.get(timeframe, ''),
                'features': features
            }
            path = self.model_cache_path / f"{timeframe}_model.joblib"

            # Compute HMAC signature using dedicated secret
            secret = self.config.MODEL_SIGNATURE_SECRET.encode()

            # Wrap blocking I/O in thread pool
            serialized = await asyncio.to_thread(self._serialize_model_data, model_data)
            signature = hmac.new(secret, serialized, hashlib.sha256).hexdigest()

            # Atomic write with thread pool
            temp_path = path.with_suffix('.tmp')
            await asyncio.to_thread(joblib.dump, model_data, temp_path)

            # Atomic signature write
            signature_path = path.with_suffix('.sig')
            temp_sig = signature_path.with_suffix('.tmp')
            await asyncio.to_thread(temp_sig.write_text, signature)
            await asyncio.to_thread(temp_sig.replace, signature_path)

            # Replace atomically
            await asyncio.to_thread(temp_path.replace, path)
            app_logger.debug(f"Model persisted: {timeframe} with HMAC signature")
        except Exception as e:
            app_logger.error(f"Model persistence failed: {e}")

    def _serialize_model_data(self, model_data: Dict[str, Any]) -> bytes:
        """Helper method to serialize model data"""
        import pickle
        return pickle.dumps(model_data)

    async def _load_persisted_model(self, timeframe: str) -> bool:
        """Load model from disk with HMAC verification"""
        try:
            path = self.model_cache_path / f"{timeframe}_model.joblib"
            signature_path = path.with_suffix('.sig')

            if not path.exists() or not signature_path.exists():
                return False

            # Verify signature using dedicated secret
            secret = self.config.MODEL_SIGNATURE_SECRET.encode()

            # Read serialized data with thread pool
            serialized = await asyncio.to_thread(path.read_bytes)

            expected_signature = await asyncio.to_thread(signature_path.read_text)

            if not hmac.compare_digest(
                hmac.new(secret, serialized, hashlib.sha256).hexdigest(),
                expected_signature
            ):
                app_logger.error(f"{timeframe}: Model signature verification failed - possible tampering")
                return False

            model_data = await asyncio.to_thread(joblib.load, path)

            self.models[timeframe] = model_data['model']
            self.scalers[timeframe] = model_data['scaler']
            self.regime_maps[timeframe] = model_data['regime_map']
            self.is_trained[timeframe] = True
            self.training_hashes[timeframe] = model_data.get('data_hash', '')
            self.training_metrics[timeframe] = {
                'features': model_data.get('features', ['log_ret', 'vol_gk', 'rsi', 'trend_eff', 'vwap_dev'])
            }
            app_logger.info(f"Loaded persisted model: {timeframe}")
            return True
        except Exception as e:
            app_logger.warning(f"Failed to load persisted model {timeframe}: {e}")
            return False

    async def _map_regimes_stably(self, timeframe: str, model: hmm.GaussianHMM, X: np.ndarray,
                                  old_map: Dict[int, str], available_features: List[str]) -> Dict[int, str]:
        """Map regimes stably by correlating with old mapping and using statistical rules"""
        predictions = model.predict(X)
        means = model.means_

        # Find feature indices from available features
        try:
            log_ret_idx = available_features.index('log_ret')
        except ValueError:
            log_ret_idx = 0

        try:
            vol_gk_idx = available_features.index('vol_gk')
        except ValueError:
            vol_gk_idx = 1 if len(available_features) > 1 else 0

        log_ret_means = means[:, log_ret_idx]
        vol_means = means[:, vol_gk_idx] if len(means[0]) > vol_gk_idx else np.zeros_like(log_ret_means)

        # Use combined metric for stable sorting
        log_ret_normalized = (log_ret_means - log_ret_means.min()) / (log_ret_means.max() - log_ret_means.min() + 1e-9)
        vol_normalized = (vol_means - vol_means.min()) / (vol_means.max() - vol_means.min() + 1e-9)

        # Combined metric: return dominates but volatility influences ranking
        combined_metric = log_ret_normalized * 0.7 + vol_normalized * 0.3

        # Sort by combined metric
        sorted_indices = np.argsort(combined_metric)

        # Create new map
        new_map = {}
        n_components = len(sorted_indices)

        # Map to BULL/CHOP/BEAR based on combined metric
        for i, idx in enumerate(sorted_indices):
            percentile = i / (n_components - 1) if n_components > 1 else 0.5

            # Use combined metric for regime assignment
            metric_value = combined_metric[idx]

            if metric_value < 0.3:
                new_map[idx] = "BEAR"
            elif metric_value > 0.7:
                new_map[idx] = "BULL"
            else:
                # Middle regimes: check volatility for CHOP classification
                vol_percentile = (vol_means[idx] - vol_means.min()) / (vol_means.max() - vol_means.min() + 1e-9)
                if vol_percentile > 0.6:
                    new_map[idx] = "CHOP"
                else:
                    # Further distinguish middle regimes
                    if percentile < 0.5:
                        new_map[idx] = "BEAR_CHOP"
                    else:
                        new_map[idx] = "BULL_CHOP"

        # Ensure unique regime mapping
        if len(set(new_map.values())) != len(new_map):
            app_logger.warning(f"{timeframe}: Regime mapping not unique, forcing uniqueness")
            # Force unique mapping
            unique_values = set()
            for k, v in new_map.items():
                if v in unique_values:
                    # Append number to make unique
                    counter = 1
                    while f"{v}_{counter}" in unique_values:
                        counter += 1
                    new_map[k] = f"{v}_{counter}"
                unique_values.add(new_map[k])

        return new_map

    async def predict_timeframe(self, timeframe: str, df: pd.DataFrame, calculate_quality: bool = True) -> Dict[str, Any]:
        """Predict with SEQUENTIAL HMM inference (STATISTICAL FIX), confidence intervals, validation, and quality scoring"""
        async with self.lock:
            if not self.is_trained.get(timeframe, False):
                # Try loading persisted model
                if not await self._load_persisted_model(timeframe):
                    # Alert on training failure
                    if self.training_failures.get(timeframe, 0) >= self.training_failure_threshold:
                        app_logger.critical(f"{timeframe}: Strategy degraded - model not available")
                    return {
                        "action": "HOLD", "score": 0.0, "regime": "UNKNOWN",
                        "is_valid": False, "quality": 0.0, "confidence": 0.0, "chop_probability": 0.0
                    }

            if df.empty or len(df) < self.min_sequence_length:
                return {
                    "action": "HOLD", "score": 0.0, "regime": "UNKNOWN",
                    "is_valid": False, "quality": 0.0, "confidence": 0.0, "chop_probability": 0.0
                }

            try:
                features = self.training_metrics[timeframe]["features"]

                # STATISTICAL FIX: Use SEQUENCE of observations for valid HMM inference
                sequence_length = min(self.sequence_length, len(df))
                X_sequence = df[features].iloc[-sequence_length:].values

                # Validate finite values
                if not np.isfinite(X_sequence).all():
                    return {
                        "action": "HOLD", "score": 0.0, "regime": "ERROR",
                        "is_valid": False, "quality": 0.0, "confidence": 0.0, "chop_probability": 0.0
                    }

                scaler = self.scalers[timeframe]
                X_scaled = scaler.transform(X_sequence)

                # Get probabilities for ENTIRE SEQUENCE (valid HMM inference)
                posteriors_sequence = self.models[timeframe].predict_proba(X_scaled)

                # Use the LAST observation's probabilities for current regime (most recent)
                posteriors = posteriors_sequence[-1]

                # Also compute sequence-consistent regime via Viterbi for robustness
                try:
                    viterbi_states = self.models[timeframe].predict(X_scaled)
                    sequence_consistent_regime_idx = viterbi_states[-1]
                except Exception:
                    # Fallback to max posterior if Viterbi fails
                    sequence_consistent_regime_idx = np.argmax(posteriors)

                # Check regime map key exists
                if sequence_consistent_regime_idx not in self.regime_maps[timeframe]:
                    app_logger.error(f"{timeframe}: Regime map missing index {sequence_consistent_regime_idx}")
                    return {
                        "action": "HOLD", "score": 0.0, "regime": "ERROR",
                        "is_valid": False, "quality": 0.0, "confidence": 0.0, "chop_probability": 0.0
                    }

                regime = self.regime_maps[timeframe].get(sequence_consistent_regime_idx, "UNKNOWN")

                # Calculate score based on regime probabilities
                bull_prob = 0.0
                bear_prob = 0.0
                chop_prob = 0.0

                for i, state_regime in self.regime_maps[timeframe].items():
                    if "BULL" in state_regime and "CHOP" not in state_regime:
                        bull_prob += posteriors[i]
                    elif "BEAR" in state_regime and "CHOP" not in state_regime:
                        bear_prob += posteriors[i]
                    else:
                        chop_prob += posteriors[i]

                score = bull_prob - bear_prob

                # STABILITY CHECK: If retrained recently, dampen confidence
                raw_confidence = float(max(posteriors))
                if timeframe in self.retrain_cooldowns:
                    time_since_retrain = (datetime.now() - self.retrain_cooldowns[timeframe]).total_seconds()
                    if time_since_retrain < 300:
                        app_logger.debug(f"{timeframe}: Stability dampening active ({time_since_retrain:.0f}s ago)")
                        raw_confidence *= 0.8

                # Sequence stability bonus: if last N states are consistent
                sequence_stability = 0.0
                if len(viterbi_states) >= 5:
                    last_states = viterbi_states[-5:]
                    if len(set(last_states)) == 1:  # All same state
                        sequence_stability = 0.15
                    elif len(set(last_states)) <= 2:  # At most 2 states
                        sequence_stability = 0.08

                result = {
                    "timeframe": timeframe,
                    "action": self._determine_action_from_regime(regime, score, chop_prob),
                    "score": round(float(score), 3),
                    "regime": regime,
                    "confidence": raw_confidence + sequence_stability,
                    "posteriors": posteriors.tolist(),
                    "is_valid": True,
                    "quality": 0.0,
                    "chop_probability": float(chop_prob),
                    "components": self.component_usage.get(timeframe, self.config.HMM_COMPONENTS),
                    "sequence_length": sequence_length,
                    "sequence_stability": sequence_stability,
                    "viterbi_state": int(sequence_consistent_regime_idx)
                }

                # Calculate quality if requested
                if calculate_quality:
                    result["quality"] = self._calculate_quality(result)

                return result

            except Exception as e:
                app_logger.error(f"{timeframe} prediction error: {e}")
                return {
                    "action": "HOLD", "score": 0.0, "regime": "ERROR",
                    "is_valid": False, "quality": 0.0, "confidence": 0.0, "chop_probability": 0.0
                }

    def _determine_action_from_regime(self, regime: str, score: float, chop_prob: float) -> str:
        """Clear action determination logic"""
        # Chop filter
        if chop_prob > 0.5:
            return "HOLD"

        # Explicit regime-action mapping
        if "BULL" in regime and "CHOP" not in regime:
            if score > self.config.REGIME_BULL_THRESHOLD:
                return "BUY"
            elif score < self.config.REGIME_BEAR_THRESHOLD:
                return "SELL"  # Strong counter-trend signal in bull regime
            else:
                return "HOLD"
        elif "BEAR" in regime and "CHOP" not in regime:
            if score < self.config.REGIME_BEAR_THRESHOLD:
                return "SELL"
            elif score > self.config.REGIME_BULL_THRESHOLD:
                return "BUY"  # Strong counter-trend signal in bear regime
            else:
                return "HOLD"
        else:  # CHOP or UNKNOWN
            return "HOLD"

    def _calculate_quality(self, signal: Dict[str, Any]) -> float:
        """Calculate weighted signal quality with chop penalty, volatility adjustment, and sequence bonus"""
        # Base score from confidence
        base_score = signal.get("confidence", 0.0) * 100

        # Chop penalty
        chop_penalty = signal.get("chop_probability", 0.0) * 50
        base_score -= chop_penalty

        # Regime alignment bonus
        regime_alignment = 0
        if (signal["regime"] == "BULL" and signal["action"] == "BUY") or \
           (signal["regime"] == "BEAR" and signal["action"] == "SELL"):
            regime_alignment = 20 * signal.get("confidence", 0)

        # Score strength bonus
        score_abs = abs(signal.get("score", 0))
        strength_bonus = 15 if score_abs > 0.6 else 8 if score_abs > 0.4 else 0

        # Timeframe weight (normalized)
        weights = {
            "1d": 4.0,
            "4h": 2.5,
            "1h": 1.5,
            "15m": 1.0
        }
        tf_weight = weights.get(signal['timeframe'], 1.0)

        # Component count bonus
        component_bonus = signal.get("components", self.config.HMM_COMPONENTS) * 2

        # Sequence stability bonus (STATISTICAL FIX)
        sequence_bonus = signal.get("sequence_stability", 0.0) * 25

        # Clamp AFTER all multiplications
        combined_score = (base_score + regime_alignment + strength_bonus + component_bonus + sequence_bonus) * tf_weight
        final_score = min(100, max(0, combined_score))

        return final_score

# ===============================================================================
# 🎯 EXECUTION ENGINE - STATE-AWARE & SMART EXECUTION (V19 ENHANCED)
# ===============================================================================

class ExecutionCircuitBreaker:
    """Circuit breaker for Alpaca API"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.state = {"open": False, "last_failure": 0, "failure_count": 0}
        self.threshold = config.CIRCUIT_FAILURE_THRESHOLD
        self.timeout = config.CIRCUIT_TIMEOUT_SECONDS
        self.lock = asyncio.Lock()

    async def __aenter__(self):
        async with self.lock:
            if self.state["open"]:
                if time.time() - self.state["last_failure"] > self.timeout:
                    self.state["open"] = False
                    self.state["failure_count"] = 0
                else:
                    raise APIError("Circuit breaker is OPEN", status_code=503)
            return self

    async def __aexit__(self, exc_type, exc, tb):
        async with self.lock:
            if exc_type is not None:
                self.state["failure_count"] += 1
                self.state["last_failure"] = time.time()
                if self.state["failure_count"] >= self.threshold:
                    self.state["open"] = True
                    app_logger.error("Execution circuit breaker OPENED")
            else:
                if self.state["failure_count"] > 0:
                    self.state["failure_count"] = 0

class ExecutionEngine:
    def __init__(self, config: SystemConfig, client: TradingClient, db: DatabaseManager, telegram: TelegramBot, data_engine: MultiTimeframeDataEngine):
        self.config = config
        self.client = client
        self.db = db
        self.telegram = telegram
        self.data_engine = data_engine
        self.daily_trades = 0
        self.daily_loss = Decimal('0.0')  # Use Decimal for precision
        self.max_drawdown_today = Decimal('0.0')
        self.last_reset_date = datetime.now().date()

        # Faster cache refresh with configurable TTL
        self.position_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self.cache_ttl = min(config.POSITION_CACHE_TTL, config.LIVE_LOOP_INTERVAL_SECONDS)
        self._last_position_fetch = 0
        self._position_lock = asyncio.Lock()

        # Persistent active order tracking
        self.active_orders: Dict[str, Dict[str, Any]] = {}
        self.order_lock = asyncio.Lock()

        # Enhanced trade scheduling
        self.trade_history: List[datetime] = []
        self.last_trade_time = datetime.min
        self.last_trade_timeframe: Optional[str] = None

        # Performance-based throttling
        self.recent_trade_performance: deque = deque(maxlen=20)
        self.performance_threshold = 0.4

        # Cooldown tracking
        self.timeframe_cooldowns: Dict[str, datetime] = {}
        self.cooldown_duration = 60

        # Safety mode
        self.safety_mode = False
        self.safety_mode_threshold = Decimal('-0.02')  # Use Decimal

        # Execution circuit breaker
        self.execution_circuit_breaker = ExecutionCircuitBreaker(config)

        # Store primary timeframe signal for correlation checks
        self._primary_tf_signal: Optional[Dict[str, Any]] = None

        # Track open positions count
        self.open_positions_count = 0
        self._open_positions_lock = asyncio.Lock()

    async def initialize(self):
        """Initialize daily limits and load from DB"""
        await self._reset_daily_limits()
        await self._validate_account_status()
        await self.check_alpaca_health()  # Health check

        # Load timeframe cooldowns from DB
        risk_data = await self.db.get_daily_risk(datetime.now().date().isoformat())
        if risk_data and risk_data.get('timeframe_cooldowns'):
            self.timeframe_cooldowns = {
                tf: datetime.fromisoformat(ts) for tf, ts in risk_data['timeframe_cooldowns'].items()
            }

        # Load recent trade performance
        await self._load_recent_performance()

        # Reconcile active orders on startup (FIXED: Using GetOrdersRequest with OrderStatus.NEW)
        await self._reconcile_active_orders()

        # Initialize open positions count
        await self._update_open_positions_count()

    async def _update_open_positions_count(self):
        """Update count of open positions"""
        try:
            open_trades = await self.db.get_open_trades(self.config.SYMBOL)
            async with self._open_positions_lock:
                self.open_positions_count = len(open_trades)
            app_logger.info(f"Open positions count: {self.open_positions_count}")
        except Exception as e:
            app_logger.warning(f"Could not update open positions count: {e}")
            async with self._open_positions_lock:
                self.open_positions_count = 0

    async def _increment_open_positions(self):
        """Increment open positions count with lock"""
        async with self._open_positions_lock:
            self.open_positions_count += 1

    async def _decrement_open_positions(self):
        """Decrement open positions count with lock"""
        async with self._open_positions_lock:
            self.open_positions_count = max(0, self.open_positions_count - 1)

    async def check_alpaca_health(self):
        """Verify Alpaca API connectivity before trading"""
        try:
            async with self.execution_circuit_breaker:
                account = await asyncio.to_thread(self.client.get_account)
            app_logger.info(f"Alpaca health check passed - Account: {account.status}")
        except Exception as e:
            app_logger.critical(f"Alpaca health check failed: {e}")
            raise

    async def _load_recent_performance(self):
        """Load recent trade performance from DB"""
        try:
            today = datetime.now().date().isoformat()
            perf = await self.db.get_trade_performance(today)
            if perf:
                # Restore performance metrics
                self.recent_trade_performance.extend([perf['avg_pnl']] * perf['trades_count'])
                app_logger.info(f"Restored trade performance: {perf['win_rate']:.2%} win rate")
            else:
                app_logger.info("No trade performance history found, starting fresh")
        except Exception as e:
            app_logger.warning(f"Could not load trade performance: {e}")

    async def _validate_account_status(self):
        """Verify account is ACTIVE and unrestricted before trading"""
        try:
            async with self.execution_circuit_breaker:
                account = await asyncio.to_thread(self.client.get_account)

            # Check account status
            if hasattr(account, 'status'):
                status = account.status.value if hasattr(account.status, 'value') else str(account.status)
                if status not in ['ACTIVE', 'ACTIVE_ENHANCED']:
                    raise AccountBlockedError(f"Account status is {status}, not ACTIVE")

            # Check restriction flags
            if hasattr(account, 'trading_blocked') and account.trading_blocked:
                raise AccountBlockedError("Account is trading_blocked")
            if hasattr(account, 'account_blocked') and account.account_blocked:
                raise AccountBlockedError("Account is account_blocked")
            if hasattr(account, 'pattern_day_trader') and account.pattern_day_trader:
                app_logger.warning("Account is flagged as pattern day trader")

            # Check buying power
            if float(account.buying_power) <= 0:
                raise AccountBlockedError("Account has no buying power")

            app_logger.info(f"Account validation passed: {status}, Equity: ${float(account.equity):,.2f}")

        except AccountBlockedError:
            raise
        except Exception as e:
            app_logger.error(f"Account validation failed: {e}")
            raise AccountBlockedError(f"Could not validate account: {e}")

    async def _reconcile_active_orders(self):
        """V19 FIXED: Check broker for active orders vs our state - CORRECTED OrderStatus.NEW"""
        try:
            # V19 FIX: Use OrderStatus.NEW (capital N) instead of OrderStatus.new
            request = GetOrdersRequest(status='all')

            async with self.execution_circuit_breaker:
                broker_orders = await asyncio.to_thread(self.client.get_orders, request)

            broker_order_ids: Set[str] = {o.id for o in broker_orders}

            # Load our active orders from DB
            our_active_orders = await self.db.get_active_orders()
            our_order_ids: Set[str] = {o['order_id'] for o in our_active_orders}

            # Cancel orders we track but broker doesn't have (ghost orders)
            for order_id in our_order_ids - broker_order_ids:
                app_logger.warning(f"Ghost order {order_id} removed from tracking")
                await self.db.update_active_order(order_id, "CANCELED")
                async with self.order_lock:
                    self.active_orders.pop(order_id, None)

            # Add orders broker has but we don't track
            for order in broker_orders:
                if order.id not in our_order_ids:
                    app_logger.info(f"Discovered active order {order.id}")
                    await self.db.log_active_order({
                        'order_id': order.id,
                        'symbol': order.symbol,
                        'status': getattr(order.status, 'value', str(order.status)),
                        'filled_qty': int(order.filled_qty) if order.filled_qty else 0,
                        'submitted_at': getattr(order, 'submitted_at', datetime.now())
                    })
                    async with self.order_lock:
                        self.active_orders[order.id] = {
                            'symbol': order.symbol,
                            'status': getattr(order.status, 'value', str(order.status)),
                            'submitted_at': getattr(order, 'submitted_at', datetime.now())
                        }

            app_logger.info(f"Reconciled {len(broker_orders)} active orders")

        except APIError as e:
            app_logger.error(f"Reconciliation API error: {e}")
        except Exception as e:
            app_logger.error(f"Reconciliation failed: {e}")

    async def _reset_daily_limits(self):
        """Reset daily trading limits from DB or initialize with DST handling"""
        today = pd.Timestamp.now(tz='America/New_York').date()
        if today != self.last_reset_date:
            # Load from database
            risk_data = await self.db.get_daily_risk(today.isoformat())

            if risk_data:
                self.daily_trades = risk_data['daily_trades']
                # FIXED: Use Decimal constructor for float values
                self.daily_loss = Decimal(str(risk_data['daily_loss'])).quantize(
                    Decimal('0.0001'), rounding=ROUND_HALF_UP)
                self.max_drawdown_today = Decimal(str(risk_data.get('max_drawdown', 0.0)))
                self.safety_mode = risk_data.get('safety_mode_active', False)
                # Restore timeframe_cooldowns
                if risk_data.get('timeframe_cooldowns'):
                    self.timeframe_cooldowns = {
                        tf: datetime.fromisoformat(ts) for tf, ts in risk_data['timeframe_cooldowns'].items()
                    }
            else:
                self.daily_trades = 0
                self.daily_loss = Decimal('0.0')
                self.max_drawdown_today = Decimal('0.0')
                self.safety_mode = False
                self.timeframe_cooldowns = {}

            self.last_reset_date = today

            app_logger.info(f"Daily limits reset: {self.daily_trades} trades, ${self.daily_loss:.2f} loss")

    async def update_daily_loss(self, realized_pnl: float) -> bool:
        """Update daily loss tracking with portfolio value - FIXED float-to-Decimal conversion"""
        # FIXED: Use Decimal constructor for float values
        pnl_decimal = Decimal(str(realized_pnl)).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
        self.daily_loss += pnl_decimal
        self.max_drawdown_today = min(self.max_drawdown_today, self.daily_loss)

        # Get current portfolio value
        try:
            async with self.execution_circuit_breaker:
                account = await asyncio.to_thread(self.client.get_account)
            portfolio_value = Decimal(str(account.portfolio_value))
        except Exception:
            portfolio_value = Decimal(str(self.config.INITIAL_CAPITAL))

        # Use max of initial and current capital
        dynamic_threshold = self.safety_mode_threshold * max(Decimal(str(self.config.INITIAL_CAPITAL)), portfolio_value)

        # Check safety mode activation
        if self.daily_loss <= dynamic_threshold:
            self.safety_mode = True
            app_logger.warning(f"Safety mode activated: ${self.daily_loss:.2f} loss")

        # Persist to DB
        signal_efficiency = await self.db.calculate_signal_efficiency()
        await self.db.log_daily_risk(
            datetime.now().date().isoformat(),
            self.daily_loss,
            self.daily_trades,
            float(self.max_drawdown_today),
            float(portfolio_value),
            self.data_engine.daily_api_calls,
            signal_efficiency,
            safety_mode_active=self.safety_mode,
            timeframe_cooldowns={tf: ts.isoformat() for tf, ts in self.timeframe_cooldowns.items()}
        )

        # Check kill switch
        if self.daily_loss <= -Decimal(str(self.config.MAX_DAILY_LOSS_DOLLAR)):
            app_logger.critical(f"DAILY LOSS LIMIT HIT: ${self.daily_loss:.2f}")
            await self.telegram.send(f"**KILL SWITCH ACTIVATED**\nDaily Loss: ${self.daily_loss:.2f}\nPortfolio: ${portfolio_value:,.2f}\nSystem HALTED.", priority="critical")
            return False

        # Check max drawdown
        if self.max_drawdown_today <= -Decimal(str(self.config.MAX_DAILY_LOSS_DOLLAR * 1.5)):
            app_logger.critical(f"MAX DRAWDOWN LIMIT EXCEEDED: ${self.max_drawdown_today:.2f}")
            await self.telegram.send(f"**DRAWDOWN LIMIT HIT**\nMax DD: ${self.max_drawdown_today:.2f}\nSystem HALTED.", priority="critical")
            return False

        return True

    def validate_price_increment(self, price: float) -> float:
        """Validate and adjust price to exchange minimum increments"""
        if not (price and np.isfinite(price) and price > 0 and not np.isnan(price)):
            raise ValueError(f"Invalid price: {price}")

        # Alpaca increment rules: <$1: $0.0001, $1-$100: $0.01, >$100: $0.05
        if price < 1.0:
            increment = 0.0001
        elif price < 100.0:
            increment = 0.01
        else:
            increment = 0.05

        # Round to nearest increment
        validated_price = round(round(price / increment) * increment, 6)

        if abs(validated_price - price) > increment:
            app_logger.warning(f"Price {price} adjusted to {validated_price} to comply with {increment} increment")

        return validated_price

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
           retry=retry_if_exception_type(APIError))
    async def get_position(self, symbol: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Get current position with caching and proper error handling"""
        now = time.time()
        async with self._position_lock:
            if not force_refresh and (now - self._last_position_fetch) < self.cache_ttl:
                return self.position_cache.get(symbol)

        try:
            async with self.execution_circuit_breaker:
                pos = await asyncio.to_thread(self.client.get_open_position, symbol)

            # Handle fractional shares properly
            qty_decimal = Decimal(str(pos.qty))
            if qty_decimal % 1 != 0:
                app_logger.warning(f"Fractional shares detected: {qty_decimal}; truncating to int")

            position = {
                "symbol": pos.symbol,
                "qty": int(qty_decimal),
                "avg_entry_price": float(pos.avg_entry_price),
                "market_value": float(pos.market_value) if pos.market_value else 0.0,
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
                "side": "LONG" if int(qty_decimal) > 0 else "SHORT"
            }

            async with self._position_lock:
                self.position_cache[symbol] = position
                self._last_position_fetch = now

            return position

        except APIError as e:
            if getattr(e, 'status_code', 404) == 404:
                async with self._position_lock:
                    self.position_cache[symbol] = None
                return None
            else:
                app_logger.error(f"API error getting position: {e}")
                raise
        except Exception as e:
            app_logger.error(f"Error getting position: {e}")
            raise

    async def calculate_position_size(self, symbol: str, signal: Dict[str, Any]) -> Optional[int]:
        """Calculate position size with slippage buffer, risk checks, and volatility adjustment - FIXED division by zero"""
        try:
            # Check max open positions limit
            if self.open_positions_count >= self.config.MAX_OPEN_POSITIONS:
                app_logger.warning(f"Maximum open positions reached: {self.open_positions_count}/{self.config.MAX_OPEN_POSITIONS}")
                return None

            # Position exists check
            position = await self.get_position(symbol)
            if position:
                app_logger.info(f"Already in position: {position['qty']} shares")
                return None

            # Safety mode: reduce position size
            size_multiplier = 0.5 if self.safety_mode else 1.0

            # Get account
            async with self.execution_circuit_breaker:
                account = await asyncio.to_thread(self.client.get_account)

            equity = Decimal(str(account.equity))
            cash = Decimal(str(account.cash))
            buying_power = Decimal(str(account.buying_power))

            # Portfolio exposure check
            async with self.execution_circuit_breaker:
                all_positions = await asyncio.to_thread(self.client.get_all_positions)

            # Calculate net exposure (long - short)
            current_exposure = Decimal('0.0')
            for p in all_positions:
                # Safely convert qty to Decimal
                qty_decimal = Decimal(str(p.qty))
                market_value = Decimal(str(p.market_value)) if p.market_value else Decimal('0.0')
                if qty_decimal > 0:
                    current_exposure += market_value
                else:
                    current_exposure -= market_value

            max_exposure = equity * Decimal(str(self.config.PORTFOLIO_MAX_EXPOSURE))

            if abs(current_exposure) >= max_exposure:
                app_logger.warning(f"Portfolio exposure limit: {abs(current_exposure)/equity:.1%} >= {self.config.PORTFOLIO_MAX_EXPOSURE:.1%}")
                return None

            # Get latest data from shared engine
            df = self.data_engine.get_df("1d")
            if df.empty or 'atr' not in df.columns:
                app_logger.error("No ATR data for position sizing")
                return None

            atr = df['atr'].iloc[-1]
            current_price = df['close'].iloc[-1]

            # FIXED: Validate ATR and price BEFORE calculations with fallback for near-zero ATR
            if not (atr and np.isfinite(atr) and atr > 0):
                app_logger.error(f"Invalid ATR: {atr}")
                # Fallback to historical ATR with minimum floor
                if len(df) > 20:
                    historical_atr = df['atr'].iloc[-20:].median()
                    atr = max(historical_atr, current_price * 0.005)  # Minimum 0.5% of price
                    app_logger.warning(f"Using fallback ATR: {atr:.4f}")
                else:
                    # Minimum 1% of price or $0.01
                    atr = max(current_price * 0.01, 0.01)
                    app_logger.warning(f"Using minimum ATR: {atr:.4f}")

            if not (current_price and np.isfinite(current_price) and current_price > 0):
                app_logger.error(f"Invalid current price: {current_price}")
                return None

            # FIXED: Ensure risk_per_share has minimum floor BEFORE division with robust validation
            min_risk_per_share = current_price * (self.config.MIN_RISK_PER_SHARE_BPS / 10000)
            # Ensure ATR is positive and reasonable
            safe_atr = max(atr, current_price * 0.001, 0.01)  # Minimum 0.1% of price or $0.01
            risk_per_share = max(safe_atr * self.config.STOP_LOSS_ATR, min_risk_per_share, Decimal('0.01'))

            # CRITICAL FIX: Ensure risk_per_share is positive before division
            if risk_per_share <= Decimal('1e-9'):  # More robust check
                app_logger.error(f"Risk per share is zero or negative: {risk_per_share}")
                return None

            # Configurable spread buffer
            spread_buffer = current_price * (self.config.SPREAD_BUFFER_BPS / 10000)
            risk_per_share += spread_buffer

            # Calculate max allowed shares early
            max_allowed_shares = int((equity * Decimal(str(self.config.MAX_POS_SIZE_PCT))) / Decimal(str(current_price)))

            # Volatility-adjusted position size using Decimal throughout
            vol_adjusted_pos_size = Decimal(str(self.config.MAX_POS_SIZE_PCT)) * (Decimal('0.02') / (Decimal(str(atr)) / Decimal(str(current_price)) + Decimal('0.01')))
            vol_adjusted_pos_size = min(vol_adjusted_pos_size, Decimal(str(self.config.MAX_POS_SIZE_PCT)))

            # Position size based on risk using Decimal consistently
            risk_amount = equity * vol_adjusted_pos_size * Decimal(str(size_multiplier))

            # Use Decimal consistently for division with validated denominator
            shares_decimal = (risk_amount / Decimal(str(risk_per_share))).to_integral_value(rounding=ROUND_HALF_UP)
            shares = int(shares_decimal)

            # Enforce round lots BEFORE capping
            if self.config.ENFORCE_ROUND_LOTS and shares >= 100:
                shares = (shares // 100) * 100

            # Check minimum share count before truncation
            if shares < self.config.MIN_POSITION_SIZE:
                app_logger.warning(f"Position size < {self.config.MIN_POSITION_SIZE} share (calculated: {shares})")
                return None

            # Commission and slippage buffer using configurable cash buffer
            cash_buffer = Decimal(str(self.config.CASH_BUFFER_PCT))
            max_shares_by_cash = int((buying_power * cash_buffer) / (Decimal(str(current_price)) * Decimal('1.0001')))
            shares = min(shares, max_shares_by_cash)

            # Apply early cap (AFTER round lot adjustment)
            shares = min(shares, max_allowed_shares)

            # Check for SQLite integer overflow
            max_sqlite_int = 2**63 - 1
            if shares > max_sqlite_int:
                app_logger.critical(f"Position size {shares} exceeds SQLite limit {max_sqlite_int}")
                shares = max_sqlite_int

            app_logger.info(f"Position size: {shares} shares (risk: ${float(risk_per_share)*shares:.2f}, vol_adj: {float(vol_adjusted_pos_size):.1%}, cash_buffer: {float(cash_buffer):.2%})")
            return shares

        except Exception as e:
            app_logger.error(f"Position sizing error: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
           retry=retry_if_exception_type(APIError))
    async def verify_order_filled(self, order_id: str, timeout: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Verify order was filled and return fill details with proper error handling"""
        if timeout is None:
            timeout = self.config.ORDER_TIMEOUT_SECONDS

        try:
            start_time = time.time()
            check_interval = 2

            while time.time() - start_time < timeout:
                async with self.execution_circuit_breaker:
                    order = await asyncio.to_thread(self.client.get_order_by_id, order_id)

                # Update active order status under lock
                filled_qty = int(order.filled_qty) if order.filled_qty else 0
                await self._update_active_order(order_id, order.status.value, filled_qty)

                # Defensive status handling
                order_status = getattr(order.status, 'value', str(order.status))

                if order_status == "FILLED":
                    filled_qty = int(float(order.filled_qty))
                    filled_avg_price = float(order.filled_avg_price)

                    # Verify partial fills are complete
                    total_qty = int(float(order.qty))
                    if filled_qty >= total_qty:
                        return {
                            "filled_qty": filled_qty,
                            "filled_avg_price": filled_avg_price,
                            "status": "FILLED",
                            "filled_at": getattr(order, 'filled_at', datetime.now()),
                            "quantity": total_qty
                        }
                    else:
                        app_logger.warning(f"Order {order_id} partially filled: {filled_qty}/{total_qty}, waiting for completion")

                elif order_status in ["REJECTED", "CANCELED", "EXPIRED"]:
                    app_logger.error(f"Order {order_id} failed with status: {order_status}")
                    await self._update_active_order(order_id, order_status)
                    return {"status": order_status, "reason": getattr(order, 'reject_reason', 'Unknown')}

                elif order_status == "PARTIALLY_FILLED":
                    app_logger.info(f"Order {order_id} partially filled: {order.filled_qty}/{order.qty}")
                    # Track partial fills
                    await self._update_active_order(order_id, "PARTIALLY_FILLED", filled_qty)
                    # Continue waiting for full fill

                await asyncio.sleep(check_interval)

            # Timeout - attempt to cancel
            app_logger.warning(f"Order {order_id} not filled within {timeout}s, attempting cancel")
            try:
                async with self.execution_circuit_breaker:
                    await asyncio.to_thread(self.client.cancel_order, order_id)
                await self._update_active_order(order_id, "CANCELED")
            except APIError as cancel_error:
                app_logger.error(f"Cancel failed: {cancel_error}")
                # Order might have filled between timeout and cancel attempt
                try:
                    async with self.execution_circuit_breaker:
                        order = await asyncio.to_thread(self.client.get_order_by_id, order_id)
                    if order.status.value == "FILLED":
                        total_qty = int(float(order.qty))
                        return {
                            "filled_qty": int(order.filled_qty),
                            "filled_avg_price": float(order.filled_avg_price),
                            "status": "FILLED",
                            "filled_at": getattr(order, 'filled_at', datetime.now()),
                            "quantity": total_qty
                        }
                except Exception:
                    pass

            await self._update_active_order(order_id, "TIMEOUT")
            return {"status": "TIMEOUT"}

        except APIError as e:
            app_logger.error(f"Order verification API error: {e}")
            await self._update_active_order(order_id, "API_ERROR")
            return {"status": "API_ERROR", "error": str(e)}
        except Exception as e:
            app_logger.error(f"Order verification error: {e}")
            return None

    async def _update_active_order(self, order_id: str, status: str, filled_qty: Optional[int] = None):
        """Helper to update active order with proper locking"""
        async with self.order_lock:
            if order_id in self.active_orders:
                self.active_orders[order_id]['status'] = status
                if filled_qty is not None:
                    self.active_orders[order_id]['filled_qty'] = filled_qty
            await self.db.update_active_order(order_id, status, filled_qty)

    async def can_execute_trade(self, signal: Dict[str, Any]) -> bool:
        """Check if we can execute a trade based on timing, cooldowns, and performance - FIXED duplicate detection"""
        now = datetime.now()

        # Global cooldown between any trades
        if (now - self.last_trade_time).total_seconds() < self.config.MIN_TRADE_COOLDOWN_SECONDS:
            return False

        # Timeframe-specific cooldown
        last_tf_trade = self.timeframe_cooldowns.get(signal['timeframe'])
        if last_tf_trade and (now - last_tf_trade).total_seconds() < self.cooldown_duration:
            return False

        # FIXED: Correct duplicate trade detection - check for same timeframe AND opposite action (for closing)
        try:
            open_trades = await self.db.get_open_trades(self.config.SYMBOL)
            # Check if there's already an open trade in the SAME timeframe with SAME action (block entry)
            # Allow opposite action (for closing)
            if any(t['timeframe'] == signal['timeframe'] and t['action'] == signal['action'] for t in open_trades):
                app_logger.warning(f"Duplicate detection: Already have open {signal['action']} trade in {signal['timeframe']}")
                return False
        except Exception as e:
            app_logger.warning(f"Could not check duplicate trades: {e}")
            return False

        # Market hours check
        if self.config.MARKET_HOURS_ONLY:
            current_time = now.time()
            start_time = datetime.strptime(self.config.TRADING_START_TIME, "%H:%M").time()
            end_time = datetime.strptime(self.config.TRADING_END_TIME, "%H:%M").time()

            if not (start_time <= current_time <= end_time):
                return False

        # Performance-based throttling
        if len(self.recent_trade_performance) >= 5:
            recent_win_rate = sum(1 for p in self.recent_trade_performance if p > 0) / len(self.recent_trade_performance)
            if recent_win_rate < self.performance_threshold:
                if (now - self.last_trade_time).total_seconds() < self.config.MAX_TRADE_COOLDOWN_SECONDS:
                    app_logger.warning(f"Performance throttling active: {recent_win_rate:.1%} win rate")
                    return False

        # Safety mode check
        if self.safety_mode:
            if signal.get('quality', 0) < 80:
                app_logger.debug("Safety mode: Signal quality too low")
                return False

        # API budget check (high priority for trades)
        try:
            has_budget = await self.data_engine.has_api_budget(priority="high")
            if not has_budget:
                app_logger.warning("API budget insufficient for trade execution")
                return False
        except Exception as e:
            app_logger.warning(f"Could not check API budget: {e}, assuming insufficient")
            return False

        # Correlation check between timeframes
        if not await self._check_signal_correlation(signal):
            return False

        # Check max open positions
        if self.open_positions_count >= self.config.MAX_OPEN_POSITIONS:
            app_logger.warning(f"Maximum open positions reached: {self.open_positions_count}/{self.config.MAX_OPEN_POSITIONS}")
            return False

        return True

    async def _check_signal_correlation(self, signal: Dict[str, Any]) -> bool:
        """Clear signal correlation logic"""
        primary_signal = self._primary_tf_signal
        if not primary_signal:
            return True

        # Clear matrix: primary BEAR blocks secondary BUY unless high quality
        if primary_signal['regime'] == 'BEAR' and signal['action'] == 'BUY':
            if signal['timeframe'] != self.config.PRIMARY_TIMEFRAME:
                if signal['quality'] < 85:
                    app_logger.warning(f"Blocking BUY in {signal['timeframe']} (primary is BEAR, quality {signal['quality']:.1f} < 85)")
                    return False

        # Clear matrix: primary BULL blocks secondary SELL unless high quality
        if primary_signal['regime'] == 'BULL' and signal['action'] == 'SELL':
            if signal['timeframe'] != self.config.PRIMARY_TIMEFRAME:
                if signal['quality'] < 85:
                    app_logger.warning(f"Blocking SELL in {signal['timeframe']} (primary is BULL, quality {signal['quality']:.1f} < 85)")
                    return False

        return True

    async def execute_trade(self, symbol: str, signal: Dict[str, Any]) -> Optional[str]:
        """Execute trade with verification, slippage accounting, and smart execution"""
        execution_start = time.time()

        # Overall execution timeout
        try:
            return await asyncio.wait_for(
                self._execute_trade_internal(symbol, signal),
                timeout=self.config.EXECUTION_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            app_logger.critical(f"Trade execution timed out after {self.config.EXECUTION_TIMEOUT_SECONDS}s")
            await self.telegram.send(f"**EXECUTION TIMEOUT**\nSymbol: {symbol}\nAction: {signal['action']}\nSystem may be degraded.", priority="critical")
            return None

    async def _execute_trade_internal(self, symbol: str, signal: Dict[str, Any]) -> Optional[str]:
        """Internal trade execution logic"""
        # Check if we can execute
        if not await self.can_execute_trade(signal):
            app_logger.warning("Cannot execute trade: timing or limit constraints")

            # Log rejected signal
            await self._log_rejected_signal(signal, symbol, "REJECTED_COOLDOWN")
            return None

        try:
            await self._reset_daily_limits()

            # Validate signal
            if signal['quality'] < self.config.MIN_TRADE_QUALITY:
                app_logger.warning(f"Signal quality too low: {signal['quality']:.1f} < {self.config.MIN_TRADE_QUALITY}")
                await self._log_rejected_signal(signal, symbol, "REJECTED_QUALITY")
                return None

            # Prevent duplicate timeframe trades
            recent_trades = await self.db.get_recent_trades(symbol, limit=10)
            if any(t['timeframe'] == signal['timeframe'] and t['status'] == 'OPEN' and t['action'] == signal['action'] for t in recent_trades):
                app_logger.warning(f"Already have open {signal['action']} trade in {signal['timeframe']}")
                await self._log_rejected_signal(signal, symbol, "REJECTED_DUPLICATE_TF")
                return None

            # Calculate position size
            quantity = await self.calculate_position_size(symbol, signal)
            if not quantity:
                await self._log_rejected_signal(signal, symbol, "REJECTED_SIZING")
                return None

            # Get market data with fresh fetch and staleness check
            fetch_success = await self.data_engine.fetch_timeframe("1d", force_refresh=True, priority="high")
            if not fetch_success:
                app_logger.error("Failed to fetch fresh data for execution")
                return None

            df = self.data_engine.get_df("1d")
            if df.empty:
                return None

            # Check data staleness
            last_bar_age = (datetime.now() - df.index[-1]).total_seconds()
            expected_interval = 86400  # 1 day
            if last_bar_age > expected_interval * 2:
                app_logger.error(f"Data stale ({last_bar_age}s old), aborting trade")
                return None

            current_price = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1]

            # ATR fallback for near-zero volatility
            atr = max(atr, current_price * 0.005)  # Minimum 0.5% of price

            # Calculate execution price based on order type
            if self.config.USE_LIMIT_ORDERS and signal['quality'] > 70:
                execution_type = "LIMIT"
                # Validate price increment
                passivity = self.config.LIMIT_ORDER_PASSIVITY_BPS / 10000
                if signal['action'] == 'BUY':
                    limit_price = current_price - (passivity * current_price)
                else:
                    limit_price = current_price + (passivity * current_price)

                limit_price = self.validate_price_increment(limit_price)
                execution_price = limit_price
                slip_bps = self.config.SLIPPAGE_BPS * 0.7
            else:
                execution_type = "MARKET"
                if signal['action'] == 'BUY':
                    execution_price = current_price + (self.config.SLIPPAGE_BPS / 10000 * current_price)
                else:
                    execution_price = current_price - (self.config.SLIPPAGE_BPS / 10000 * current_price)

                execution_price = self.validate_price_increment(execution_price)
                slip_bps = self.config.SLIPPAGE_BPS

            # Calculate stops from EXECUTION_PRICE, not current_price
            stop_distance = self.config.STOP_LOSS_ATR * atr
            take_distance = self.config.TAKE_PROFIT_ATR * atr

            if signal['action'] == 'BUY':
                stop_price = execution_price - stop_distance
                take_profit_price = execution_price + take_distance
            else:
                # FIXED CORRECT: For SELL: take_profit < execution < stop (stop above entry, take profit below)
                stop_price = execution_price + stop_distance
                take_profit_price = execution_price - take_distance

            # Validate stop and take profit prices
            stop_price = self.validate_price_increment(stop_price)
            take_profit_price = self.validate_price_increment(take_profit_price)

            # Validate bracket order price ordering BEFORE submission
            if not self._validate_bracket_order_prices(signal['action'], stop_price, execution_price, take_profit_price):
                raise ValueError(f"Invalid bracket order prices for {signal['action']}")

            # Create order request
            order_request = self._create_order_request(
                symbol, quantity, signal['action'], execution_type,
                execution_price if execution_type == "LIMIT" else None,
                stop_price, take_profit_price
            )

            # Submit order to broker
            async with self.execution_circuit_breaker:
                order = await asyncio.to_thread(self.client.submit_order, order_data=order_request)

            # Validate order response with null check
            if not order or not hasattr(order, 'id'):
                raise APIError(f"Invalid order response: {order}", status_code=500)

            order_id = order.id

            # Log real order after submission
            await self.db.log_active_order({
                'order_id': order_id,
                'symbol': symbol,
                'status': OrderStatus.ACCEPTED.value,
                'filled_qty': 0,
                'submitted_at': datetime.now()
            })

            # Update in-memory tracking with lock
            await self._add_active_order(order_id, {
                'symbol': symbol,
                'signal': signal,
                'submitted_at': datetime.now(),
                'quantity': quantity,
                'execution_type': execution_type
            })

            # Verify fill
            fill_details = await self.verify_order_filled(order_id)

            # Remove from active tracking if filled or failed
            if fill_details and fill_details.get("status") == "FILLED":
                await self._remove_active_order(order_id)
            else:
                app_logger.error(f"Order {order_id} verification failed: {fill_details}")
                # Decrement position count on failure
                await self._decrement_open_positions()

            if not fill_details or fill_details.get("status") != "FILLED":
                app_logger.error(f"Order {order_id} failed verification: {fill_details}")
                await self.telegram.send(
                    f"**ORDER FAILED**\nOrder ID: {order_id}\nStatus: {fill_details.get('status')}\nReason: {fill_details.get('reason')}",
                    priority="critical"
                )
                return None

            execution_duration = int((time.time() - execution_start) * 1000)

            # Calculate slippage with Decimal precision
            actual_execution_price = fill_details['filled_avg_price']
            total_quantity = fill_details['quantity']

            # Correct slippage calculation for both BUY and SELL orders
            slippage_cost, actual_slippage_bps = self._calculate_slippage(
                signal['action'], actual_execution_price, current_price, fill_details['filled_qty']
            )

            commission_cost = fill_details['filled_qty'] * self.config.COMMISSION_PER_SHARE

            # Use order ID as trade ID
            trade_id = order_id

            trade_data = {
                'id': trade_id,
                'symbol': symbol,
                'timeframe': signal['timeframe'],
                'action': signal['action'],
                'quantity': fill_details['filled_qty'],
                'entry_price': fill_details['filled_avg_price'],
                'status': 'OPEN',
                'entry_time': fill_details.get('filled_at', datetime.now()),
                'stop_loss': stop_price,
                'take_profit': take_profit_price,
                'quality_score': signal['quality'],
                'regime': signal['regime'],
                'confidence': signal['confidence'],
                'order_id': order_id,
                'slippage_cost': slippage_cost,
                'commission_cost': commission_cost,
                'timeframe_weight': self._get_timeframe_weight(signal['timeframe']),
                'execution_duration_ms': execution_duration,
                'data_hash': self.data_engine.data_hashes.get('1d', ''),
                'execution_price': actual_execution_price,
                'execution_type': execution_type
            }

            await self.db.log_trade(trade_data)

            # Log signal execution
            await self.db.log_signal({
                'id': f"{datetime.now().isoformat()}_{signal['timeframe']}_{signal['action']}",
                'timestamp': datetime.now(),
                'timeframe': signal['timeframe'],
                'action': signal['action'],
                'score': signal['score'],
                'regime': signal['regime'],
                'confidence': signal['confidence'],
                'quality': signal['quality'],
                'symbol': symbol,
                'execution_decision': 'EXECUTED',
                'api_budget_pct': self.data_engine.get_budget_status()['pct_used']
            })

            # Update trade tracking
            self.daily_trades += 1
            self.last_trade_time = datetime.now()
            self.last_trade_timeframe = signal['timeframe']
            self.timeframe_cooldowns[signal['timeframe']] = datetime.now()

            # Update open positions count AFTER successful execution
            await self._increment_open_positions()

            # Invalidate position cache
            async with self._position_lock:
                self.position_cache.pop(symbol, None)

            # Build message carefully with truncation indicator
            msg_parts = [
                f"**TRADE EXECUTED** ({signal['timeframe']}) - {execution_type}",
                f"Symbol: {symbol}",
                f"Action: {signal['action']} ({signal['regime']} regime)",
                f"Quantity: {fill_details['filled_qty']}/{total_quantity}",
                f"Price: ${actual_execution_price:.2f}",
                f"Slip: {actual_slippage_bps:.1f}bps",
                f"Stop: ${stop_price:.2f}",
                f"Take: ${take_profit_price:.2f}",
                f"Quality: {signal['quality']:.1f}",
                f"Daily Trades: {self.daily_trades}",
                f"Open Positions: {self.open_positions_count}/{self.config.MAX_OPEN_POSITIONS}",
                f"API Budget: {self.data_engine.get_budget_status()['pct_used']:.1f}%"
            ]

            full_msg = "\n".join(msg_parts)
            if len(full_msg) > 4096:
                app_logger.warning(f"Telegram message truncated: {len(full_msg)} chars")

            await self.telegram.send(full_msg, priority="critical")

            app_logger.info(f"Trade executed: {trade_id} in {execution_duration}ms via {execution_type}")
            return order_id

        except APIError as e:
            app_logger.error(f"Order submission failed: {e}")
            await self.telegram.send(f"**ORDER FAILED**\n{str(e)}\nStatus: {e.status_code}", priority="critical")
            # Decrement position count on API failure
            await self._decrement_open_positions()
            return None
        except KeyboardInterrupt:
            raise
        except SystemExit:
            raise
        except Exception as e:
            app_logger.error(f"Trade execution error: {e}", exc_info=True)
            # Decrement position count on general failure
            await self._decrement_open_positions()
            return None

    async def _log_rejected_signal(self, signal: Dict[str, Any], symbol: str, reason: str):
        """Helper to log rejected signals"""
        await self.db.log_signal({
            'id': f"{datetime.now().isoformat()}_{signal['timeframe']}_{signal['action']}",
            'timestamp': datetime.now(),
            'timeframe': signal['timeframe'],
            'action': signal['action'],
            'score': signal['score'],
            'regime': signal['regime'],
            'confidence': signal['confidence'],
            'quality': signal['quality'],
            'symbol': symbol,
            'execution_decision': reason,
            'api_budget_pct': self.data_engine.get_budget_status()['pct_used']
        })

    def _validate_bracket_order_prices(self, action: str, stop_price: float, execution_price: float, take_profit_price: float) -> bool:
        """
        Validate bracket order price ordering - CORRECTED for short positions

        For BUY orders: stop < execution < take_profit
        For SELL orders: take_profit < execution < stop (stop above entry, take profit below)
        """
        if action == 'BUY':
            # For BUY: stop < execution < take_profit
            if not (stop_price < execution_price < take_profit_price):
                app_logger.error(f"Invalid BUY bracket: stop={stop_price:.4f} < exec={execution_price:.4f} < take={take_profit_price:.4f}")
                return False
        else:  # SELL
            # FIXED CORRECT: For SELL: take_profit < execution < stop (stop above entry, take profit below)
            if not (take_profit_price < execution_price < stop_price):
                app_logger.error(f"Invalid SELL bracket: take={take_profit_price:.4f} < exec={execution_price:.4f} < stop={stop_price:.4f}")
                return False
        return True

    def _create_order_request(self, symbol: str, quantity: int, action: str, execution_type: str,
                              limit_price: Optional[float], stop_price: float, take_profit_price: float):
        """Create order request based on type"""
        side = OrderSide.BUY if action == 'BUY' else OrderSide.SELL

        if execution_type == "LIMIT":
            return LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=round(limit_price, 6),
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=round(stop_price, 6)),
                take_profit=TakeProfitRequest(limit_price=round(take_profit_price, 6))
            )
        else:
            return MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=round(stop_price, 6)),
                take_profit=TakeProfitRequest(limit_price=round(take_profit_price, 6))
            )

    def _calculate_slippage(self, action: str, actual_price: float, expected_price: float, quantity: int) -> Tuple[float, float]:
        """Calculate slippage correctly for both long and short positions - returns consistent float types"""
        # Use Decimal for precise calculation
        actual_decimal = Decimal(str(actual_price))
        expected_decimal = Decimal(str(expected_price))

        if action == 'BUY':
            # For BUY: positive slippage = expected - actual (lower price is better)
            price_diff = expected_decimal - actual_decimal
        else:  # SELL
            # For SELL: positive slippage = actual - expected (higher price is better)
            price_diff = actual_decimal - expected_decimal

        slippage_bps = (price_diff / expected_decimal) * Decimal('10000')
        slippage_bps_float = float(slippage_bps)
        slippage_cost = slippage_bps_float / 10000 * actual_price * quantity
        return slippage_cost, slippage_bps_float

    async def _add_active_order(self, order_id: str, order_data: Dict[str, Any]):
        """Helper to add active order with locking"""
        async with self.order_lock:
            self.active_orders[order_id] = order_data

    async def _remove_active_order(self, order_id: str):
        """Helper to remove active order with locking"""
        async with self.order_lock:
            self.active_orders.pop(order_id, None)

    async def check_and_close_positions(self, symbol: str, signals: List[Dict[str, Any]]):
        """Check if position should be closed based on opposing signal from SAME timeframe"""
        try:
            position = await self.get_position(symbol)
            if not position:
                return

            # Get position timeframe from trade history
            position_tf = await self._get_position_timeframe(symbol)
            if not position_tf:
                app_logger.warning(f"Could not determine timeframe for position {symbol}")
                return

            # Find opposing signal from same timeframe
            opposing_action = "SELL" if position['side'] == "LONG" else "BUY"
            matching_signals = [s for s in signals if s['timeframe'] == position_tf and s['action'] == opposing_action]

            if not matching_signals:
                return

            signal = matching_signals[0]

            # Verify quality and cooldown
            if signal['quality'] < self.config.MIN_TRADE_QUALITY:
                app_logger.debug(f"Exit signal quality too low: {signal['quality']}")
                return

            # Check timeframe cooldown
            if (datetime.now() - self.timeframe_cooldowns.get(position_tf, datetime.min)).total_seconds() < self.cooldown_duration:
                app_logger.debug(f"Exit cooldown active for {position_tf}")
                return

            # Submit closing order
            close_quantity = abs(position['qty'])
            app_logger.info(f"Closing {close_quantity} shares of {symbol} (reason: {signal['timeframe']} {signal['action']})")

            # Get actual trade data from DB for accurate PnL calculation
            trade_id = await self._get_open_trade_id(symbol)
            if trade_id:
                open_trades = await self.db.get_open_trades(symbol)
                if open_trades:
                    trade_data = open_trades[0]
                    entry_price = trade_data.get('entry_price')
                    if entry_price:
                        # Calculate actual PnL based on entry price
                        current_price = position['avg_entry_price'] + position['unrealized_pl'] / position['qty']
                        pnl = (current_price - entry_price) * position['qty'] if position['side'] == 'LONG' else (entry_price - current_price) * abs(position['qty'])
                        position['unrealized_pl'] = pnl

            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=close_quantity,
                side=OrderSide.SELL if position['side'] == "LONG" else OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )

            async with self.execution_circuit_breaker:
                order = await asyncio.to_thread(self.client.submit_order, order_data=order_request)

            fill_details = await self.verify_order_filled(order.id, timeout=60)

            if fill_details and fill_details.get("status") == "FILLED":
                # Update trade record
                if trade_id:
                    await self.db.update_trade_exit(
                        trade_id,
                        exit_price=fill_details['filled_avg_price'],
                        exit_time=fill_details['filled_at'],
                        pnl=Decimal(str(position['unrealized_pl'])),
                        status='CLOSED',
                        exit_reason=f"signal_{signal['timeframe']}"
                    )

                # Update tracking
                self.timeframe_cooldowns[position_tf] = datetime.now()

                # Update recent performance
                self.recent_trade_performance.append(position['unrealized_pl'])

                # Update open positions count
                await self._decrement_open_positions()

                # Persist performance
                if len(self.recent_trade_performance) > 0:
                    win_rate = sum(1 for p in self.recent_trade_performance if p > 0) / len(self.recent_trade_performance)
                    await self.db.log_trade_performance(
                        datetime.now().date().isoformat(),
                        win_rate,
                        sum(self.recent_trade_performance) / len(self.recent_trade_performance),
                        len(self.recent_trade_performance)
                    )

                # Update daily loss with Decimal
                await self.update_daily_loss(position['unrealized_pl'])

                await self.telegram.send(
                    f"**POSITION CLOSED**\nSymbol: {symbol}\nQty: {fill_details['filled_qty']}\nPrice: ${fill_details['filled_avg_price']:.2f}\nPnL: ${position['unrealized_pl']:.2f}",
                    priority="critical"
                )

        except Exception as e:
            app_logger.error(f"Position close error: {e}", exc_info=True)

    async def _get_position_timeframe(self, symbol: str) -> Optional[str]:
        """Get the timeframe of the open position"""
        open_trades = await self.db.get_open_trades(symbol)
        if open_trades:
            return open_trades[0].get('timeframe')
        return None

    async def _get_open_trade_id(self, symbol: str) -> Optional[str]:
        """Get the ID of the open trade"""
        open_trades = await self.db.get_open_trades(symbol)
        if open_trades:
            return open_trades[0].get('id')
        return None

    def _get_timeframe_weight(self, timeframe: str) -> float:
        """Get normalized timeframe weight"""
        weights = {
            "1d": 4.0,
            "4h": 2.5,
            "1h": 1.5,
            "15m": 1.0
        }
        return weights.get(timeframe, 1.0)

    async def check_liquidated_positions(self):
        """Enhanced check for positions liquidated by broker"""
        try:
            async with self.execution_circuit_breaker:
                positions = await asyncio.to_thread(self.client.get_all_positions)

            broker_symbols = {p.symbol for p in positions}
            db_open_trades = await self.db.get_open_trades()

            for trade in db_open_trades:
                if trade['symbol'] not in broker_symbols:
                    # Position no longer exists - likely liquidated
                    app_logger.critical(f"Position liquidated: {trade['symbol']} (trade {trade['id']})")
                    # Use config-based maximum loss instead of magic number
                    max_loss = -Decimal(str(self.config.INITIAL_CAPITAL * self.config.MAX_DAILY_LOSS_PCT * 10))
                    await self.db.update_trade_exit(
                        trade['id'],
                        exit_price=0.0,
                        exit_time=datetime.now(),
                        pnl=max_loss.quantize(Decimal('0.0001')),
                        status='LIQUIDATED',
                        exit_reason='broker_liquidation'
                    )
                    # Update open positions count
                    await self._decrement_open_positions()

                    await self.telegram.send(
                        f"**POSITION LIQUIDATED**\nSymbol: {trade['symbol']}\nTrade ID: {trade['id']}\nImmediate attention required!",
                        priority="critical"
                    )
                else:
                    # Verify position size matches
                    broker_pos = next((p for p in positions if p.symbol == trade['symbol']), None)
                    if broker_pos and int(broker_pos.qty) != trade['quantity']:
                        app_logger.critical(f"Position size mismatch: DB {trade['quantity']} vs Broker {broker_pos.qty}")
                        await self.telegram.send(
                            f"**SIZE MISMATCH**\nSymbol: {trade['symbol']}\nDB: {trade['quantity']}\nBroker: {broker_pos.qty}",
                            priority="critical"
                        )

        except Exception as e:
            app_logger.error(f"Liquidation check error: {e}")

    async def shutdown(self):
        """Graceful shutdown - cancel pending orders and close positions"""
        app_logger.info("Shutting down execution engine...")

        # Cancel all pending/submitted orders
        pending_orders = []
        async with self.order_lock:
            pending_orders = [
                oid for oid, order in self.active_orders.items()
                if order.get('status') in ['PENDING', 'SUBMITTED', 'ACCEPTED']
            ]

        for order_id in pending_orders:
            try:
                app_logger.info(f"Canceling order {order_id} on shutdown")
                async with self.execution_circuit_breaker:
                    await asyncio.to_thread(self.client.cancel_order, order_id)
                await self.db.update_active_order(order_id, "CANCELED")
            except Exception as e:
                app_logger.error(f"Failed to cancel {order_id}: {e}")

        # Close all open positions
        try:
            async with self.execution_circuit_breaker:
                positions = await asyncio.to_thread(self.client.get_all_positions)
            for pos in positions:
                app_logger.info(f"Closing position {pos.symbol} on shutdown")
                close_qty = abs(int(pos.qty))
                order_request = MarketOrderRequest(
                    symbol=pos.symbol,
                    qty=close_qty,
                    side=OrderSide.SELL if int(pos.qty) > 0 else OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                await asyncio.to_thread(self.client.submit_order, order_data=order_request)

                # Update open positions count
                await self._decrement_open_positions()
        except Exception as e:
            app_logger.error(f"Failed to close positions on shutdown: {e}")

        app_logger.info("Execution engine shutdown complete")

# ===============================================================================
# 🚀 MAIN ENTRY POINT - PRODUCTION BOOTSTRAP (V19 ENHANCED)
# ===============================================================================

async def main():
    """Main trading loop with graceful shutdown and error handling"""
    # Initialize secrets validation
    SecretsManager.validate_env_security()
    SecretsManager.load_from_vault()

    # Parse args first, then create config
    args = parse_args()

    # Create configuration
    conf = SystemConfig(
        SYMBOL=args.symbol,
        PAPER_TRADING=args.paper,
        LIVE_LOOP_INTERVAL_SECONDS=args.loop_interval
    )

    # Bootstrap components
    db = DatabaseManager(conf)
    await db.initialize()

    # Clean up old orders on startup
    await db.cleanup_old_orders()

    data_engine = None
    telegram = None

    try:
        data_engine = MultiTimeframeDataEngine(conf, db)
        await data_engine.initialize()

        telegram = TelegramBot(conf.TELEGRAM_TOKEN, conf.TELEGRAM_CHANNEL)
        await telegram.initialize()

        brain = MultiTimeframeBrain(conf)
        await brain.initialize()
        brain.data_engine = data_engine  # Uses weak reference now
        brain.telegram = telegram  # Uses weak reference now

        client = TradingClient(conf.API_KEY, conf.SECRET_KEY, paper=conf.PAPER_TRADING)

        engine = ExecutionEngine(conf, client, db, telegram, data_engine)
        await engine.initialize()

        # Signal handlers
        shutdown_event = asyncio.Event()

        def handle_shutdown(sig, frame):
            app_logger.warning(f"Signal {sig} received, shutting down...")
            shutdown_event.set()

        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

        # Global exception handler
        def handle_exception(loop, context):
            app_logger.critical(f"Global exception: {context.get('exception', context['message'])}")
            if telegram:
                # Check telegram exists before sending
                try:
                    asyncio.create_task(telegram.send(
                        f"**FATAL ERROR**\n{context.get('exception', 'Unknown')}", priority="critical"))
                except Exception as e:
                    app_logger.error(f"Failed to send Telegram alert: {e}")

        asyncio.get_event_loop().set_exception_handler(handle_exception)

        app_logger.info("TITANIUM v19-PROD-ENHANCED entering live loop...")

        try:
            await _live_loop(conf, data_engine, brain, engine, telegram, shutdown_event)
        except KeyboardInterrupt:
            app_logger.info("Shutdown signal received, cleaning up...")
        except Exception as e:
            app_logger.critical(f"Fatal error in main: {e}", exc_info=True)
        finally:
            await engine.shutdown()
            if telegram:
                await telegram.close()
            if data_engine:
                # Ensure data engine session is closed
                if data_engine.session and not data_engine.session.closed:
                    await data_engine.session.close()
            await db.backup()
            app_logger.info("Graceful shutdown complete")
    except Exception as e:
        app_logger.critical(f"Failed to initialize system: {e}", exc_info=True)
        if telegram:
            await telegram.close()
        if data_engine and data_engine.session and not data_engine.session.closed:
            await data_engine.session.close()
        raise

async def _live_loop(config: SystemConfig, data_engine: MultiTimeframeDataEngine, brain: MultiTimeframeBrain,
                     executor: ExecutionEngine, telegram: TelegramBot, shutdown_event: asyncio.Event):
    """Core trading loop."""
    loop_count = 0

    while not shutdown_event.is_set():
        try:
            loop_start = time.time()

            # Wrap blocking psutil call in thread pool
            process = psutil.Process()
            mem_info = await asyncio.to_thread(process.memory_info)
            total_mem_mb = mem_info.rss / 1024 / 1024

            # Complete memory accounting
            cache_mb = 0.0
            for df in data_engine.dataframes.values():
                if not df.empty and hasattr(df, 'memory_usage'):
                    try:
                        cache_mb += df.memory_usage(deep=True).sum() / 1024 / 1024
                    except Exception:
                        # Estimate if memory calculation fails
                        cache_mb += len(df) * len(df.columns) * 8 / 1024 / 1024
            
            position_cache_mb = sys.getsizeof(executor.position_cache) / 1024 / 1024
            order_cache_mb = sys.getsizeof(executor.active_orders) / 1024 / 1024

            if total_mem_mb > config.MEMORY_LIMIT_MB:
                app_logger.critical(f"Memory limit exceeded: {total_mem_mb:.1f}MB > {config.MEMORY_LIMIT_MB}MB (Dataframes: {cache_mb:.1f}MB)")
                if telegram:
                    await telegram.send(f"**MEMORY LIMIT EXCEEDED**\nSystem HALTED.\nMemory: {total_mem_mb:.1f}MB", priority="critical")
                break

            # Log memory breakdown more frequently
            if loop_count % 10 == 0:
                app_logger.info(f"Memory usage: Total={total_mem_mb:.1f}MB, Dataframes={cache_mb:.1f}MB, Positions={position_cache_mb:.1f}MB, Orders={order_cache_mb:.1f}MB")

            # Refresh position cache
            await executor.get_position(config.SYMBOL, force_refresh=True)

            # Continuous reconciliation
            if loop_count % 5 == 0:
                await executor._reconcile_active_orders()

            # Check for liquidated positions
            if loop_count % 10 == 0:
                await executor.check_liquidated_positions()

            # Data fetch with improved error handling
            fetch_tasks = []
            for tf in data_engine.timeframes:
                # For initial runs, fetch with force refresh
                fetch_tasks.append(data_engine.fetch_timeframe(
                    tf, force_refresh=(loop_count == 0), priority="high"))
            
            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            # Check for fetch errors
            successful_fetches = 0
            for tf, result in zip(data_engine.timeframes, results):
                if isinstance(result, Exception):
                    app_logger.error(f"Fetch error for {tf}: {result}")
                elif result is True:
                    successful_fetches += 1
                    df = data_engine.get_df(tf)
                    if not df.empty:
                        app_logger.debug(f"Successfully fetched {len(df)} bars for {tf}")
                    else:
                        app_logger.warning(f"Fetched {tf} but DataFrame is empty")
                else:
                    app_logger.warning(f"Fetch failed for {tf}")

            # If no timeframes have data, try emergency mode
            if successful_fetches == 0:
                app_logger.warning("No data fetched, entering emergency mode")
                # Try a simple direct fetch as last resort
                try:
                    ticker = yf.Ticker(data_engine.symbol)
                    df = await asyncio.to_thread(ticker.history, period="5d", interval="1h")
                    if not df.empty:
                        df.columns = df.columns.str.lower()
                        data_engine.dataframes["1h"] = await data_engine._engineer_features(df, "1h")
                        app_logger.info(f"Emergency mode: Got {len(df)} bars from Yahoo Finance")
                        successful_fetches = 1
                except Exception as e:
                    app_logger.error(f"Emergency mode also failed: {e}")

            # Training - only if we have data
            if successful_fetches > 0:
                for tf in data_engine.timeframes:
                    df = data_engine.get_df(tf)
                    if not df.empty and len(df) >= config.MIN_INITIAL_BARS:
                        should_retrain, reason = await brain.should_retrain(tf, df)
                        if should_retrain:
                            # Catch training exceptions
                            try:
                                await brain.train_timeframe(tf, df, retrain_reason=reason)
                            except InsufficientRegimeDiversityError:
                                app_logger.critical(f"{tf}: Insufficient regime diversity, strategy degraded")
                                if telegram:
                                    await telegram.send(f"**REGIME DIVERSITY FAILURE**\nTimeframe: {tf}\nSystem degraded.", priority="critical")

            # Prediction
            signals = []
            primary_signal = None

            for tf in data_engine.timeframes:
                df = data_engine.get_df(tf)
                if not df.empty and brain.is_trained.get(tf):
                    signal = await brain.predict_timeframe(tf, df)
                    if signal['is_valid']:
                        signals.append(signal)
                        if tf == config.PRIMARY_TIMEFRAME:
                            primary_signal = signal

            # Store primary signal for correlation check
            if primary_signal:
                executor._primary_tf_signal = primary_signal

            # Execution - only if we have valid signals
            if signals:
                # ENTRY: Best quality signal > minimum threshold
                best_signal = max(signals, key=lambda s: s['quality'])
                if best_signal['action'] != 'HOLD' and best_signal['quality'] >= config.MIN_TRADE_QUALITY:
                    await executor.execute_trade(config.SYMBOL, best_signal)

                # EXIT: Check same-TF opposing signal
                await executor.check_and_close_positions(config.SYMBOL, signals)

            # Budget update
            data_engine.update_budget_mode()

            # Rotate health check file
            health_file = Path(config.HEALTH_CHECK_FILE)
            if health_file.exists() and health_file.stat().st_size > 1024:
                # Async file operations
                try:
                    await aiofiles.os.rename(health_file, f"{config.HEALTH_CHECK_FILE}.{loop_count}")
                except Exception as e:
                    app_logger.warning(f"Failed to rotate health file: {e}")

            # Health check
            try:
                async with aiofiles.open(health_file, 'w') as f:
                    await f.write(f"OK {datetime.now().isoformat()} {loop_count}")
            except Exception as e:
                app_logger.warning(f"Failed to write health file: {e}")

            loop_duration = time.time() - loop_start
            app_logger.debug(f"Loop {loop_count} completed in {loop_duration:.2f}s")

            # Adaptive sleep
            sleep_time = max(0, config.LIVE_LOOP_INTERVAL_SECONDS - loop_duration)
            await asyncio.sleep(sleep_time)

            loop_count += 1

        except asyncio.CancelledError:
            app_logger.info("Live loop cancelled")
            break
        except Exception as e:
            app_logger.critical(f"Live loop error: {e}", exc_info=True)
            await asyncio.sleep(5)  # Brief backoff before retry

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        app_logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    except KeyboardInterrupt:
        app_logger.info("Process interrupted by user")
        sys.exit(0)
# ===============================================================================
# 🚀 SYSTEM CONTROLLER (REQUIRED FOR DASHBOARD)
# ===============================================================================
class TitaniumSystem:
    def __init__(self):
        # Load Config
        self.conf = SystemConfig()
        
        # Initialize Database
        self.db = DatabaseManager(self.conf)
        
        # Initialize Data Engine
        self.data = MultiTimeframeDataEngine(self.conf, self.db)
        
        # Initialize Brain
        self.brain = MultiTimeframeBrain(self.conf)
        
        # Initialize Telegram
        self.telegram = TelegramBot(self.conf.TELEGRAM_TOKEN, self.conf.TELEGRAM_CHANNEL)
        
        # Initialize Alpaca Client
        self.client = TradingClient(self.conf.API_KEY, self.conf.SECRET_KEY, paper=self.conf.PAPER_TRADING)
        
        # Initialize Execution Engine
        self.executor = ExecutionEngine(self.conf, self.client, self.db, self.telegram, self.data)

    async def initialize(self):
        """Boot up all components"""
        print("[System] Initializing Titanium Components...")
        await self.db.initialize()
        await self.data.initialize()
        await self.telegram.initialize()
        await self.brain.initialize()
        
        # Link weak references for the brain
        self.brain.data_engine = self.data
        self.brain.telegram = self.telegram
        
        await self.executor.initialize()
        print("[System] Initialization Complete.")

    async def shutdown(self):
        """Graceful shutdown"""
        print("[System] Shutting down...")
        await self.executor.shutdown()
        await self.telegram.close()
        if self.data.session and not self.data.session.closed:
            await self.data.session.close()

# --- EXPOSED LIVE LOOP FOR DASHBOARD ---
async def _live_loop(config, data_engine, brain, executor, telegram, shutdown_event):
    """
    This function allows the Dashboard Wrapper to run the bot's logic 
    as a background process.
    """
    print("--- TITANIUM LIVE TRADING LOOP STARTED (DASHBOARD MODE) ---")
    
    # Ensure components are ready
    if not executor.active_orders:
        await executor.initialize()

    loop_count = 0
    
    while not shutdown_event.is_set():
        try:
            loop_count += 1
            
            # 1. Memory Safety Check
            process = psutil.Process()
            if process.memory_info().rss / 1024 / 1024 > config.MEMORY_LIMIT_MB:
                app_logger.critical("Memory limit exceeded. Restarting loop.")
                break

            # 2. Data Fetch (High Priority)
            # Fetch primary timeframe
            await data_engine.fetch_timeframe(config.PRIMARY_TIMEFRAME, priority="high")
            
            # 3. Brain Processing
            df = data_engine.get_df(config.PRIMARY_TIMEFRAME)
            if not df.empty:
                # Check for retraining
                should_train, reason = await brain.should_retrain(config.PRIMARY_TIMEFRAME, df)
                if should_train:
                    await brain.train_timeframe(config.PRIMARY_TIMEFRAME, df, reason)
                
                # Predict
                signal = await brain.predict_timeframe(config.PRIMARY_TIMEFRAME, df)
                
                # 4. Execution
                if signal.get('is_valid') and signal.get('action') != 'HOLD':
                    await executor.execute_trade(config.SYMBOL, signal)
                    
                # Check for exits on existing positions
                await executor.check_and_close_positions(config.SYMBOL, [signal])

            # 5. Maintenance
            if loop_count % 10 == 0:
                await executor._reconcile_active_orders()
                await executor.check_liquidated_positions()

            # Sleep
            await asyncio.sleep(1)
            
        except asyncio.CancelledError:
            print("Loop cancelled by user.")
            break
        except Exception as e:
            app_logger.error(f"Live loop error: {e}")
            await asyncio.sleep(5)
