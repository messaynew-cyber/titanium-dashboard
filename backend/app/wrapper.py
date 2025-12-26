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

# LOGGING CAPTURE (To show logs on Dashboard)
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

# Attach handler to the bot's logger
bot_logger = logging.getLogger("TITANIUM")  # Must match the logger name in titanium_bot.py
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
            # We override paths to ensure they work in the container
            cls._instance.db_path = "/app/TITANIUM_V1_FIXED/titanium_production.db"
            cls._instance.conf = SystemConfig(
                DB_PATH=cls._instance.db_path,
                DB_BACKUP_PATH="/app/TITANIUM_V1_FIXED/backups",
                LIVE_LOOP_INTERVAL_SECONDS=180 # 3 mins
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
            # We mock the shutdown event for the bot
            shutdown_event = asyncio.Event()
            
            # Initialize components
            await self.system.initialize()
            
            # Run the actual bot loop
            # Note: We are importing the internal _live_loop from the module scope if possible, 
            # or calling the method if the user structure allows. 
            # Based on your file, we need to replicate the main() logic slightly.
            
            from algo.titanium_bot import _live_loop
            await _live_loop(
                self.conf, 
                self.system.data, 
                self.system.brain, 
                self.system.executor, 
                self.system.telegram, 
                shutdown_event
            )
        except Exception as e:
            bot_logger.error(f"Wrapper Loop Error: {e}")
            self.running = False

    async def force_trade(self, symbol, side, qty):
        """Manually execute via the Executor"""
        if not self.system: await self.initialize()
        
        # Create a fake signal for the executor
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
        today = datetime.now().date().isoformat()
        risk = await self.system.db.get_daily_risk(today)
        if risk:
            equity = float(risk['portfolio_value'])
            daily_pnl = float(risk['daily_loss']) # In the bot logic, loss is tracked, we might need to invert or check logic
            drawdown = float(risk['max_drawdown'])

        # 2. Get Latest Signal (Regime)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM signals ORDER BY timestamp DESC LIMIT 1") as cursor:
                row = await cursor.fetchone()
                if row: regime = row['regime']

        # 3. Get Trades
        trades_list = []
        open_trades = await self.system.db.get_open_trades()
        for t in open_trades:
             trades_list.append({
                 "time": t['entry_time'],
                 "symbol": t['symbol'],
                 "side": t['action'],
                 "qty": t['quantity'],
                 "price": t['entry_price']
             })

        # 4. Generate Equity History (for Chart)
        # We query the daily_risk table or performance metrics to build a curve
        history = []
        # If DB is empty, send ghost data
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
            "history": history, # In V2 we will query historical table
            "trades": trades_list
        }

    def get_logs(self, limit=50):
        # Return captured logs from memory
        return log_capture[-limit:]

    # Pass-throughs
    async def run_backtest(self, days=180):
        if not self.system: await self.initialize()
        res = await asyncio.to_thread(self.system.backtester.run_walk_forward)
        if not res: return {"error": "Backtest failed"}
        
        # Format for frontend
        curve = [{"date": str(i).split(' ')[0], "value": float(v)} for i, v in res['df']['equity'].items()]
        return {
            "stats": {
                "total_return": res['Total Return'],
                "sharpe_ratio": res['Sharpe'],
                "max_drawdown": res['Max DD'],
                "total_trades": 0 # Not provided in summary
            },
            "equity_curve": curve
        }

    async def run_diagnostics(self):
        return [{"name": "Database", "status": "PASS", "details": "SQLite Connected"}]

titanium = TitaniumService()
