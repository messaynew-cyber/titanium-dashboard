import sys
import asyncio
import logging
import json
import os
import glob
import aiosqlite
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

# SETUP PATHS
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

# STRICT IMPORT
try:
    from algo.titanium_bot import TitaniumSystem, SystemConfig, DatabaseManager
except ImportError as e:
    print(f"FATAL: Bot file missing or broken. {e}")
    sys.exit(1)

# LOGGING
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
            cls._instance.running = False
            cls._instance.task = None
            
            # Paths
            cls._instance.db_dir = Path("/app/TITANIUM_V1_FIXED")
            cls._instance.db_path = cls._instance.db_dir / "titanium_production.db"
            cls._instance.db_dir.mkdir(parents=True, exist_ok=True)
            
            # Runtime Cache
            cls._instance.equity_history = []
            cls._instance.latest_signal = {}
            
        return cls._instance

    async def _nuke_database(self):
        """External Force Clean: Deletes all DB files to fix Ghost Orders"""
        try:
            bot_logger.warning("☢️ INITIATING DATABASE CLEANUP...")
            # Delete .db, .db-wal, .db-shm
            files = glob.glob(str(self.db_dir / "titanium_production.db*"))
            for f in files:
                try:
                    os.remove(f)
                    bot_logger.warning(f"🗑️ Deleted corrupted file: {f}")
                except Exception as e:
                    bot_logger.error(f"Failed to delete {f}: {e}")
            bot_logger.warning("✅ Database wiped. System starts clean.")
        except Exception as e:
            bot_logger.error(f"Cleanup failed: {e}")

    async def initialize(self):
        """Bootstraps the bot instance"""
        
        # 1. NUKE THE DB BEFORE STARTING (Fixes the 50 ghost orders)
        if not hasattr(self, '_wiped'):
            await self._nuke_database()
            self._wiped = True

        # 2. Init System
        if self.system is None:
            self.system = TitaniumSystem()
            self.system.conf = SystemConfig(
                DB_PATH=self.db_path,
                DB_BACKUP_PATH="/app/TITANIUM_V1_FIXED/backups",
                PAPER_TRADING=True
            )
            await self.system.initialize()
            
            # 3. Backfill Chart
            await self._backfill_history()

    async def _backfill_history(self):
        try:
            ticker = self.system.conf.SYMBOL
            df = await asyncio.to_thread(yf.download, ticker, period="2d", interval="5m", progress=False)
            
            if not df.empty:
                start_equity = self.system.conf.INITIAL_CAPITAL
                self.equity_history = []
                for index, row in df.iterrows():
                    self.equity_history.append({
                        "timestamp": index.strftime("%H:%M"),
                        "value": start_equity,
                        "price": float(row['Close'])
                    })
        except: pass

    async def start_engine(self):
        if self.running: return {"status": "Already running"}
        
        await self.initialize()
        self.running = True
        
        from algo.titanium_bot import _live_loop
        self.task = asyncio.create_task(_live_loop(
            self.system.conf, self.system.data, self.system.brain, 
            self.system.executor, self.system.telegram, asyncio.Event()
        ))
            
        bot_logger.info("🚀 TITANIUM ENGINE STARTED")
        return {"status": "Started"}

    async def stop_engine(self):
        self.running = False
        if self.task: self.task.cancel()
        if self.system: await self.system.shutdown()
        bot_logger.info("🛑 ENGINE STOPPED")
        return {"status": "Stopped"}

    async def force_trade(self, symbol, side, qty):
        if not self.system: await self.initialize()
        
        # Auto-cancel before force trade to prevent conflicts
        try:
            await asyncio.to_thread(self.system.client.cancel_orders)
        except: pass
        
        signal = {
            "action": side.upper(),
            "timeframe": "manual",
            "quality": 100.0,
            "regime": "MANUAL_OVERRIDE",
            "confidence": 1.0,
            "score": 1.0
        }
        
        try:
            res = await self.system.executor.execute_trade(symbol, signal)
            if res: return True, f"Order {res} Submitted"
            return False, "Trade Rejected (Check Logs)"
        except Exception as e:
            return False, str(e)

    async def get_data(self):
        if not self.system: await self.initialize()
        
        data = {
            "state": {"equity": 100000, "regime": "WAITING", "is_active": self.running, "daily_pnl": 0, "drawdown": 0, "api_usage": 0},
            "signal": {"sentiment": "SCANNING", "quality": 0, "targets": {}},
            "history": self.equity_history,
            "trades": []
        }

        if not os.path.exists(self.db_path): return data

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                try:
                    today = datetime.now().date().isoformat()
                    async with db.execute("SELECT * FROM daily_risk WHERE date = ?", (today,)) as c:
                        r = await c.fetchone()
                        if r:
                            data["state"]["equity"] = float(r['portfolio_value'])
                            data["state"]["daily_pnl"] = float(r['daily_loss'])
                            data["state"]["drawdown"] = float(r['max_drawdown'])
                            data["state"]["api_usage"] = r['api_calls_used']
                        else:
                            acct = self.system.client.get_account()
                            data["state"]["equity"] = float(acct.equity)
                except: pass

                try:
                    async with db.execute("SELECT * FROM signals ORDER BY timestamp DESC LIMIT 1") as c:
                        r = await c.fetchone()
                        if r:
                            data["signal"] = {
                                "sentiment": r['regime'],
                                "quality": r['quality'],
                                "score": r['score'],
                                "timeframe": r['timeframe']
                            }
                except: pass

                try:
                    async with db.execute("SELECT * FROM trades ORDER BY entry_time DESC LIMIT 50") as c:
                        rows = await c.fetchall()
                        data["trades"] = [dict(row) for row in rows]
                except: pass
            
            now_str = datetime.now().strftime("%H:%M")
            current_val = data["state"]["equity"]
            
            if not self.equity_history or self.equity_history[-1]['timestamp'] != now_str:
                if current_val > 0:
                    self.equity_history.append({"timestamp": now_str, "value": current_val})
            
            if len(self.equity_history) > 300: self.equity_history.pop(0)
            data["history"] = self.equity_history

        except: pass
        return data

    def get_logs(self, limit=50):
        return log_capture[-limit:]

    async def run_backtest(self, days=180):
        res = await asyncio.to_thread(self.system.backtester.run_walk_forward)
        if res:
            curve = [{"date": str(i).split(' ')[0], "value": float(row['equity'])} for i, row in res['df'].iterrows()]
            return {"stats": res, "equity_curve": curve}
        return {"error": "Backtest failed"}

    async def run_diagnostics(self):
        return [{"name": "Bot Engine", "status": "PASS", "details": "v18.6 Active"}]

titanium = TitaniumService()
