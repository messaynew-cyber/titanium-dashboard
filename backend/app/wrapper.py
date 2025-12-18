import sys
import threading
import time
import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
import os

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
        payload = {"chat_id": TG_CHANNEL_ID, "text": message, "parse_mode": "HTML"}
        requests.post(url, json=payload, timeout=5)
    except: pass

class TitaniumService:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TitaniumService, cls).__new__(cls)
            cls._instance.orchestrator = TitaniumOrchestrator()
            cls._instance.running = False
            cls._instance.thread = None
            
            # MEMORY SYSTEM
            cls._instance.history_file = Path(CFG.STATE_PATH) / "history.json"
            cls._instance.equity_history = cls._instance.load_history()
            cls._instance.trade_history = []
            cls._instance.latest_signal = {"sentiment": "SCANNING", "targets": {"entry": 0, "sl": 0, "tp": 0}}
            
        return cls._instance

    def load_history(self):
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    if len(data) > 1: return data[-500:]
            except: pass
        return [{"timestamp": datetime.now().strftime("%H:%M"), "value": 100000, "price": 0}]

    def save_history(self):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.equity_history, f)
        except: pass

    def start_engine(self):
        if self.running: return {"status": "Running"}
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        send_telegram("🚀 <b>TITANIUM X ENGAGED</b>")
        return {"status": "Started"}

    def stop_engine(self):
        self.running = False
        send_telegram("🛑 <b>TITANIUM X HALTED</b>")
        return {"status": "Stopped"}

    def run_backtest(self, days=180):
        try:
            df_symbol = self.orchestrator.data_engine.get_data(CFG.SYMBOL, days)
            df_bench = self.orchestrator.data_engine.get_data(CFG.BENCHMARK, days)
            df_feats = self.orchestrator.feature_engine.create_features(df_symbol)
            self.orchestrator.hmm_detector.train(df_feats, self.orchestrator.feature_engine.feature_cols)
            results = self.orchestrator._run_enhanced_simulation(df_feats, df_bench)
            return {
                "stats": results['stats'],
                "equity_curve": [{"date": str(d).split(' ')[0], "value": float(v)} for d, v in results['equity'].items()]
            }
        except Exception as e: return {"error": str(e)}

    def run_diagnostics(self):
        checks = []
        try:
            d = self.orchestrator.data_engine.get_data("SPY", 10)
            checks.append({"name": "Market Data", "status": "PASS", "details": f"{len(d)} bars"})
        except: checks.append({"name": "Market Data", "status": "FAIL"})
        return checks

    def _run_loop(self):
        self.orchestrator._run_startup_checks()
        CFG.MAX_POSITION_SIZE, CFG.MAX_GROSS_EXPOSURE = 0.40, 0.95
        RISK_MANAGER.cfg = CFG
        
        while self.running:
            try:
                price = self.orchestrator.execution_engine.get_price(CFG.SYMBOL)
                qty, _, _ = self.orchestrator.execution_engine.get_position(CFG.SYMBOL)
                acct = self.orchestrator.execution_engine.get_account_info()
                cash = float(acct.get('cash', 0))
                real_equity = cash + (qty * price)
                
                # Update Permanent History
                now_str = datetime.now().strftime("%H:%M")
                if not self.equity_history or self.equity_history[-1]['timestamp'] != now_str:
                    self.equity_history.append({"timestamp": now_str, "value": real_equity, "price": price})
                else:
                    self.equity_history[-1] = {"timestamp": now_str, "value": real_equity, "price": price}
                
                if len(self.equity_history) > 500: self.equity_history.pop(0)
                self.save_history()
                
                STATE.update(equity=real_equity, cash=cash, position_qty=qty)

                # TRADING LOGIC
                df = self.orchestrator.data_engine.get_data(CFG.SYMBOL, 365)
                if not df.empty:
                    df_f = self.orchestrator.feature_engine.create_features(df)
                    self.orchestrator.hmm_detector.train(df_f, self.orchestrator.feature_engine.feature_cols)
                    regime = self.orchestrator.hmm_detector.predict(df_f, self.orchestrator.feature_engine.feature_cols)
                    signal = self.orchestrator.strategy.generate_signal(df_f, regime)
                    
                    # ELITE SIGNAL CALC
                    atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
                    direction = 1 if signal['signal'] >= 0 else -1
                    self.latest_signal = {
                        "sentiment": "BULLISH" if signal['signal'] > 0 else "BEARISH",
                        "targets": {"entry": price, "sl": price - (2*atr*direction), "tp": price + (4*atr*direction)},
                        "reason": f"Regime {regime['regime']} confirmed."
                    }

                    # EXECUTE
                    target_qty = int((real_equity * max(min(signal['position_pct'], 0.40), -0.40)) / price)
                    delta = target_qty - qty
                    if abs(delta * price) > 200:
                        success, oid = self.orchestrator.execution_engine.submit_order(CFG.SYMBOL, abs(delta), "buy" if delta > 0 else "sell")
                        if success:
                            send_telegram(f"✅ <b>TRADE</b>: {CFG.SYMBOL} x {abs(delta)}")
                            self.trade_history.insert(0, {"time": now_str, "symbol": CFG.SYMBOL, "side": "BUY" if delta > 0 else "SELL", "qty": abs(delta), "price": price})

                for _ in range(CFG.POLL_INTERVAL):
                    if not self.running: break
                    time.sleep(1)
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(10)

    def get_data(self):
        STATE.load()
        return {
            "state": {"equity": STATE.state.get('equity', 0), "regime": STATE.state.get('current_regime', 1), "is_active": self.running, "drawdown": STATE.state.get('current_drawdown', 0)},
            "signal": self.latest_signal, "history": self.equity_history, "trades": self.trade_history
        }

    def get_logs(self, limit=50):
        try:
            files = list(Path(CFG.LOG_PATH).glob("*.jsonl"))
            with open(max(files, key=lambda f: f.stat().st_mtime), 'r') as f:
                return [json.loads(l) for l in f.readlines()[-limit:]]
        except: return []

    def force_trade(self, s, side, q):
        return self.orchestrator.execution_engine.submit_order(s, q, side)

titanium = TitaniumService()
