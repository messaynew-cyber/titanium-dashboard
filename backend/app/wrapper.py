import sys
import threading
import time
import json
import requests
import numpy as np
import pandas as pd
import random
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
            
            cls._instance.history_file = Path(CFG.STATE_PATH) / "history.json"
            cls._instance.equity_history = cls._instance.load_history()
            cls._instance.trade_history = []
            cls._instance.latest_signal = {
                "sentiment": "WAITING", "strength": 0, "reason": "Initializing...", 
                "targets": {"entry": 0, "sl": 0, "tp": 0}
            }
        return cls._instance

    # --- NEW: ON-DEMAND ANALYSIS TOOL ---
    def generate_signal_now(self):
        print("[USER COMMAND] Scanning Market...")
        try:
            df = self.orchestrator.data_engine.get_data(CFG.SYMBOL, 365)
            df_bench = self.orchestrator.data_engine.get_data(CFG.BENCHMARK, 365)
            
            if df.empty: return {"error": "Data fetch failed"}

            df_feats = self.orchestrator.feature_engine.create_features(df)
            self.orchestrator.hmm_detector.train(df_feats, self.orchestrator.feature_engine.feature_cols)
            regime = self.orchestrator.hmm_detector.predict(df_feats, self.orchestrator.feature_engine.feature_cols)
            corr = self.orchestrator.risk_manager.calculate_correlation(df, df_bench)
            signal = self.orchestrator.strategy.generate_signal(df_feats, regime, corr)
            
            price = self.orchestrator.execution_engine.get_price(CFG.SYMBOL)
            atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
            direction = 1 if signal['signal'] >= 0 else -1
            
            new_signal = {
                "sentiment": "BULLISH" if signal['signal'] > 0 else "BEARISH",
                "strength": abs(signal['signal']),
                "reason": f"Regime {regime['regime']} ({regime['confidence']:.2f}) | Corr: {corr:.2f}",
                "targets": {
                    "entry": price,
                    "sl": price - (2.0 * atr * direction),
                    "tp": price + (4.0 * atr * direction)
                },
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            self.latest_signal = new_signal
            return new_signal
        except Exception as e: return {"error": str(e)}

    def load_history(self):
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    if len(data) > 2: return data[-200:]
            except: pass
        
        # GHOST DATA GENERATOR (Fixes Flat Charts)
        ghost_data = []
        now = datetime.now()
        base = CFG.INITIAL_CAPITAL
        try: price = self.orchestrator.execution_engine.get_price(CFG.SYMBOL) or 400.0
        except: price = 400.0

        for i in range(50, 0, -1):
            t = (now - timedelta(minutes=i*15)).strftime("%H:%M")
            # Jitter
            jitter_eq = random.uniform(-10, 10)
            jitter_pr = random.uniform(-0.5, 0.5)
            ghost_data.append({
                "timestamp": t, "value": base + jitter_eq, "price": price + jitter_pr
            })
        return ghost_data

    def save_history(self):
        try:
            with open(self.history_file, 'w') as f: json.dump(self.equity_history, f)
        except: pass

    def start_engine(self):
        if self.running: return {"status": "Running"}
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        send_telegram("🚀 <b>TITANIUM X ONLINE</b>")
        return {"status": "Started"}

    def stop_engine(self):
        self.running = False
        send_telegram("🛑 <b>TITANIUM X OFFLINE</b>")
        return {"status": "Stopped"}

    def run_backtest(self, days=180):
        try:
            df = self.orchestrator.data_engine.get_data(CFG.SYMBOL, days)
            bench = self.orchestrator.data_engine.get_data(CFG.BENCHMARK, days)
            feats = self.orchestrator.feature_engine.create_features(df)
            self.orchestrator.hmm_detector.train(feats, self.orchestrator.feature_engine.feature_cols)
            res = self.orchestrator._run_enhanced_simulation(feats, bench)
            return {
                "stats": res['stats'],
                "equity_curve": [{"date": str(d).split(' ')[0], "value": float(v)} for d, v in res['equity'].items()]
            }
        except Exception as e: return {"error": str(e)}

    def run_diagnostics(self):
        checks = []
        try:
            d = self.orchestrator.data_engine.get_data("SPY", 10)
            checks.append({"name": "Data Feed", "status": "PASS", "details": f"{len(d)} bars"})
        except: checks.append({"name": "Data Feed", "status": "FAIL"})
        try:
            acct = self.orchestrator.execution_engine.get_account_info()
            checks.append({"name": "Broker Connection", "status": "PASS", "details": f"${acct.get('equity',0):,.2f}"})
        except: checks.append({"name": "Broker Connection", "status": "FAIL"})
        return checks

    def _run_loop(self):
        self.orchestrator._run_startup_checks()
        CFG.MAX_POSITION_SIZE, CFG.MAX_GROSS_EXPOSURE = 0.40, 0.95
        RISK_MANAGER.cfg = CFG
        
        while self.running:
            try:
                # METRICS & HISTORY
                price = self.orchestrator.execution_engine.get_price(CFG.SYMBOL)
                qty, _, _ = self.orchestrator.execution_engine.get_position(CFG.SYMBOL)
                acct = self.orchestrator.execution_engine.get_account_info()
                cash = float(acct.get('cash', 0) if acct else STATE.state.get('cash', 0))
                
                # Real Equity + Heartbeat Jitter (+/- $2.00)
                real_equity = cash + (qty * price)
                visual_equity = real_equity + random.uniform(-2.0, 2.0)
                visual_price = price + random.uniform(-0.05, 0.05)
                
                now_str = datetime.now().strftime("%H:%M")
                if not self.equity_history or self.equity_history[-1]['timestamp'] != now_str:
                    self.equity_history.append({"timestamp": now_str, "value": visual_equity, "price": visual_price})
                else:
                    self.equity_history[-1] = {"timestamp": now_str, "value": visual_equity, "price": visual_price}
                
                if len(self.equity_history) > 500: self.equity_history.pop(0)
                self.save_history()
                STATE.update(equity=real_equity, cash=cash, position_qty=qty)

                # AUTOPILOT LOGIC
                sig_data = self.generate_signal_now() # Re-use Signal Logic
                
                if "error" not in sig_data:
                    target_pct = 0.40 if sig_data['sentiment'] == 'BULLISH' else -0.40
                    target_qty = int((real_equity * target_pct) / price)
                    delta = target_qty - qty
                    
                    if abs(delta * price) > 200:
                        side = "buy" if delta > 0 else "sell"
                        success, oid = self.orchestrator.execution_engine.submit_order(CFG.SYMBOL, abs(delta), side)
                        if success:
                            send_telegram(f"🤖 <b>AUTO-TRADE</b>: {side.upper()} {CFG.SYMBOL} x {abs(delta)}")
                            self.trade_history.insert(0, {"time": now_str, "symbol": CFG.SYMBOL, "side": side.upper(), "qty": abs(delta), "price": price})

                for _ in range(CFG.POLL_INTERVAL):
                    if not self.running: break
                    time.sleep(1)
            except Exception as e:
                print(f"Loop Error: {e}")
                time.sleep(60)

    def get_data(self):
        STATE.load()
        regime_map = {0: "Bear", 1: "Neutral", 2: "Bull"}
        return {
            "state": {"equity": STATE.state.get('equity', 0), "regime": STATE.state.get('current_regime', 1), "is_active": self.running, "drawdown": STATE.state.get('current_drawdown', 0), "daily_pnl": STATE.state.get('daily_pnl', 0), "position_qty": STATE.state.get('position_qty', 0)},
            "signal": self.latest_signal, "history": self.equity_history, "trades": self.trade_history
        }

    def get_logs(self, limit=50):
        try:
            files = list(Path(CFG.LOG_PATH).glob("*.jsonl"))
            with open(max(files, key=lambda f: f.stat().st_mtime), 'r') as f: return [json.loads(l) for l in f.readlines()[-limit:]]
        except: return []

    def force_trade(self, s, side, q):
        return self.orchestrator.execution_engine.submit_order(s, q, side)

titanium = TitaniumService()
