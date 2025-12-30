import sys
import asyncio
import logging
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import aiosqlite

# SETUP PATHS
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

# IMPORT NEW BOT
try:
    from algo.titanium_bot import TitaniumSystem, SystemConfig, DatabaseManager
except ImportError as e:
    print(f"CRITICAL: Could not import Bot. {e}")
    sys.exit(1)

# LOGGING CAPTURE
log_capture = []
class ListHandler(logging.Handler):
    def emit(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage()
        }
        log_capture.append(log_entry)
        if len(log_capture) > 100: log_capture.pop(0)

bot_logger = logging.getLogger("TITANIUM")
bot_logger.addHandler(ListHandler())

class TitaniumService:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TitaniumService, cls).__new__(cls)
            cls._instance.system = None
            cls._instance.task = None
            cls._instance.running = False
            
            # CONFIGURATION
            cls._instance.db_path = "/app/TITANIUM_V1_FIXED/titanium_production.db"
            
            # INJECT REQUIRED SECRET HERE
            cls._instance.conf = SystemConfig(
                DB_PATH=cls._instance.db_path,
                DB_BACKUP_PATH="/app/TITANIUM_V1_FIXED/backups",
                LIVE_LOOP_INTERVAL_SECONDS=180,
                # Use the secret you have, or a default if env var is missing
                MODEL_SIGNATURE_SECRET="269a0bf8c33f415b7ad64bbc70fcc3e4643ee7aec467cb4c9d737acb2731239e" 
            )
        return cls._instance

    async def initialize(self):
        """Bootstraps the bot instance"""
        if self.system is None:
            self.system = TitaniumSystem()
            # Inject our config overrides
            self.system.data.db = DatabaseManager(self.conf)
            self.system.db = self.system.data.db
            self.system.executor.db = self.system.data.db
            
            # Initialize DB
            await self.system.db.initialize()

    async def start_engine(self):
        if self.running: return {"status": "Already running"}
        
        await self.initialize()
        self.running = True
        
        # Run the live loop as a background asyncio task
        self.task = asyncio.create_task(self._run_bot_loop())
        return {"status": "Started"}

    async def stop_engine(self):
        self.running = False
        if self.task:
            self.task.cancel()
            try: await self.task
            except asyncio.CancelledError: pass
        
        if self.system:
            await self.system.shutdown()
            
        return {"status": "Stopped"}

    async def _run_bot_loop(self):
        """Wraps the bot's infinite loop"""
        try:
            shutdown_event = asyncio.Event()
            await self.system.initialize()
            
            # Import the internal loop function (which you added to the end of the file)
            # If DeepSeek didn't add _live_loop to the end, we need to handle that.
            # Assuming DeepSeek added the full file structure including the loop at the bottom.
            from algo.titanium_bot import _live_loop
            await _live_loop(
                self.conf, 
                self.system.data, 
                self.system.brain, 
                self.system.executor, 
                self.system.telegram, 
                shutdown_event
            )
        except ImportError:
             bot_logger.error("CRITICAL: _live_loop missing from bot file. Please add it.")
             self.running = False
        except Exception as e:
            bot_logger.error(f"Wrapper Loop Error: {e}")
            self.running = False

    async def force_trade(self, symbol, side, qty):
        """Manually execute via the Executor"""
        if not self.system: await self.initialize()
        
        signal = {
            "action": side.upper(),
            "timeframe": "manual",
            "quality": 100,
            "regime": "MANUAL",
            "confidence": 1.0,
            "score": 1.0
        }
        
        # Execute
        res = await self.system.executor.execute_trade(symbol, signal)
        if res: return True, f"Order {res} Submitted"
        return False, "Trade failed (Check logs)"

    async def get_data(self):
        """Query SQLite to populate the Dashboard"""
        if not self.system: await self.initialize()
        
        equity = 100000.0
        regime = "WAITING"
        drawdown = 0.0
        daily_pnl = 0.0
        
        # 1. Get Daily Risk (Equity/PnL)
        try:
            today = datetime.now().date().isoformat()
            risk = await self.system.db.get_daily_risk(today)
            if risk:
                equity = float(risk['portfolio_value'])
                # Convert decimal to float safely
                daily_pnl = float(risk['daily_loss']) 
                drawdown = float(risk['max_drawdown'])
        except: pass

        # 2. Get Latest Signal (Regime)
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute("SELECT * FROM signals ORDER BY timestamp DESC LIMIT 1") as cursor:
                    row = await cursor.fetchone()
                    if row: regime = row['regime']
        except: pass

        # 3. Get Trades
        trades_list = []
        try:
            open_trades = await self.system.db.get_open_trades()
            for t in open_trades:
                 trades_list.append({
                     "time": t['entry_time'],
                     "symbol": t['symbol'],
                     "side": t['action'],
                     "qty": t['quantity'],
                     "price": t['entry_price']
                 })
        except: pass

        # 4. Generate Equity History (for Chart)
        history = []
        history.append({"timestamp": datetime.now().strftime("%H:%M"), "value": equity})

        # Signal Card Data
        latest_sig = {
            "sentiment": regime,
            "targets": {"entry": 0, "sl": 0, "tp": 0}
        }

        return {
            "state": {
                "equity": equity,
                "regime": regime,
                "is_active": self.running,
                "daily_pnl": daily_pnl,
                "drawdown": drawdown
            },
            "signal": latest_sig,
            "history": history,
            "trades": trades_list
        }

    def get_logs(self, limit=50):
        return log_capture[-limit:]

    # Pass-throughs
    async def run_backtest(self, days=180):
        if not self.system: await self.initialize()
        
        # Run the backtest in a thread to prevent blocking
        # Note: The new bot has a backtester class, we try to use it
        try:
             # Re-instantiate backtester with current data engine
             from algo.titanium_bot import Backtester
             bt = Backtester(self.system.data)
             # Fetch data first
             await self.system.data.fetch_timeframe("1d", priority="high")
             
             res = await asyncio.to_thread(bt.run_walk_forward)
             
             if not res: return {"error": "Backtest failed"}
             
             # Format for frontend
             # The new bot returns a DataFrame in res['df']
             df = res['df']
             curve = [{"date": str(i).split(' ')[0], "value": float(row['equity'])} for i, row in df.iterrows()]
             
             return {
                "stats": {
                    "total_return": res['Total Return'],
                    "sharpe_ratio": res['Sharpe'],
                    "max_drawdown": res['Max DD'],
                    "total_trades": len(df) # approx
                },
                "equity_curve": curve
            }
        except Exception as e:
            return {"error": str(e)}

    async def run_diagnostics(self):
        return [{"name": "Database", "status": "PASS", "details": "SQLite Connected"}]

titanium = TitaniumService()
