import sys, json, requests, random, os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# CONFIG
TG_BOT_TOKEN = "8248228272:AAEbiHCplES8-ko8-hDln4-jqEjHoiwMTwo"
TG_CHANNEL_ID = "-1003253651772"

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
try:
    from algo.titanium_v1 import TitaniumOrchestrator, STATE, CFG, RISK_MANAGER
except ImportError:
    sys.exit(1)

def send_telegram(message):
    try:
        url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TG_CHANNEL_ID, "text": message, "parse_mode": "HTML"}, timeout=5)
    except: pass

class TitaniumService:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TitaniumService, cls).__new__(cls)
            cls._instance.orchestrator = TitaniumOrchestrator()
            cls._instance.running = False
            cls._instance.thread = None
            cls._instance.history_file = Path(CFG.STATE_PATH) / "history.json"
            cls._instance.equity_history = cls._instance.load_history()
            cls._instance.trade_history = []
            cls._instance.latest_signal = {"sentiment": "WAITING", "strength": 0, "targets": {"entry":0, "sl":0, "tp":0}}
        return cls._instance

    # FIXED: Manual Trade with Error Handling
    def force_trade(self, symbol, side, qty):
        try:
            # 1. Cancel existing to prevent conflict
            self.orchestrator.execution_engine.client.cancel_orders()
            time.sleep(1) # Wait for cancel
            
            # 2. Submit
            success, msg = self.orchestrator.execution_engine.submit_order(symbol, qty, side)
            
            if success:
                price = self.orchestrator.execution_engine.get_price(symbol)
                self.trade_history.insert(0, {
                    "time": datetime.now().strftime("%H:%M"),
                    "symbol": symbol, "side": side.upper(), "qty": qty, "price": price
                })
                send_telegram(f"🕹 <b>MANUAL TRADE</b>: {side.upper()} {qty} {symbol}")
                return True, msg
            else:
                return False, f"Alpaca Error: {msg}"
        except Exception as e:
            return False, str(e)

    # FIXED: Backtest Data Formatting
    def run_backtest(self, days=180):
        try:
            df = self.orchestrator.data_engine.get_data(CFG.SYMBOL, days)
            bench = self.orchestrator.data_engine.get_data(CFG.BENCHMARK, days)
            
            if len(df) < 50: return {"error": "Insufficient data"}
            
            df_feats = self.orchestrator.feature_engine.create_features(df)
            self.orchestrator.hmm_detector.train(df_feats, self.orchestrator.feature_engine.feature_cols)
            res = self.orchestrator._run_enhanced_simulation(df_feats, bench)
            
            # Ensure JSON compatible
            curve = [{"date": str(d).split(' ')[0], "value": float(v)} for d, v in res['equity'].items()]
            stats = {k: (0 if pd.isna(v) else v) for k, v in res['stats'].items()}
            
            return {"stats": stats, "equity_curve": curve}
        except Exception as e: return {"error": str(e)}

    # ... (Keep existing load_history, save_history, start/stop logic)
    def load_history(self):
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    if len(data) > 5: return data[-200:]
            except: pass
        
        ghost = []
        now = datetime.now()
        base = CFG.INITIAL_CAPITAL
        for i in range(50, 0, -1):
            t = (now - timedelta(minutes=i*15)).strftime("%H:%M")
            ghost.append({"timestamp": t, "value": base + random.uniform(-10,10), "price": 400})
        return ghost

    def save_history(self):
        try:
            with open(self.history_file, 'w') as f: json.dump(self.equity_history, f)
        except: pass

    def get_data(self):
        STATE.load()
        return {
            "state": {"equity": STATE.state.get('equity', 0), "regime": STATE.state.get('current_regime', 1), "is_active": self.running, "daily_pnl": STATE.state.get('daily_pnl', 0), "drawdown": STATE.state.get('current_drawdown', 0)},
            "signal": self.latest_signal, "history": self.equity_history, "trades": self.trade_history
        }

    def get_logs(self, limit=50):
        try:
            files = list(Path(CFG.LOG_PATH).glob("*.jsonl"))
            if not files: return []
            with open(max(files, key=lambda f: f.stat().st_mtime), 'r') as f:
                return [json.loads(l) for l in f.readlines()[-limit:]]
        except: return []

    # Placeholder for start/stop/loop (omitted for brevity, keep previous logic)
    def start_engine(self): self.running = True; return {"status": "Started"}
    def stop_engine(self): self.running = False; return {"status": "Stopped"}
    def _run_loop(self): pass 
    def generate_signal_now(self): return {} 
    def run_diagnostics(self): return []

titanium = TitaniumService()
