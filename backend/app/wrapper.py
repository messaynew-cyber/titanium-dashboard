import sys
import threading
import time
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import os

# CONFIG
TG_BOT_TOKEN = "8248228272:AAEbiHCplES8-ko8-hDln4-jqEjHoiwMTwo"
TG_CHANNEL_ID = "-1003253651772"

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
try:
    from algo.titanium_v1 import TitaniumOrchestrator, STATE, CFG, RISK_MANAGER
except ImportError as e:
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
            
            # HISTORY PERSISTENCE
            cls._instance.history_file = Path(CFG.STATE_PATH) / "history.json"
            cls._instance.equity_history = cls._instance.load_history()
            
            cls._instance.trade_history = []
            cls._instance.latest_signal = {
                "sentiment": "WAITING", "reason": "Initializing...", "atr": 0,
                "targets": {"sl": 0, "tp": 0, "entry": 0}
            }
        return cls._instance

    def load_history(self):
        # 1. Try to load from disk
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    if len(data) > 2: return data[-200:]
            except: pass
        
        # 2. If empty, generate 10 "Ghost Points" (Flat line)
        # This ensures the graph ALWAYS has something to draw on startup
        ghost_history = []
        now = datetime.now()
        start_equity = CFG.INITIAL_CAPITAL
        # Try to get current price if available, else 0
        try: 
            start_price = self.orchestrator.execution_engine.get_price(CFG.SYMBOL)
        except: 
            start_price = 0

        for i in range(10, 0, -1):
            t = (now - timedelta(minutes=i*5)).strftime("%H:%M")
            ghost_history.append({
                "timestamp": t,
                "value": start_equity,
                "price": start_price
            })
        return ghost_history

    def save_history(self):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.equity_history, f)
        except: pass

    def start_engine(self):
        if self.running: return {"status": "Already running"}
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        send_telegram("🚀 <b>TITANIUM 3D STARTED</b>")
        return {"status": "Started"}

    def stop_engine(self):
        self.running = False
        send_telegram("🛑 <b>TITANIUM STOPPED</b>")
        return {"status": "Stopping..."}

    def run_backtest(self, days=180):
        try:
            df_symbol = self.orchestrator.data_engine.get_data(CFG.SYMBOL, days)
            df_bench = self.orchestrator.data_engine.get_data(CFG.BENCHMARK, days)
            if df_symbol.empty: return {"error": "No data"}
            
            df_features = self.orchestrator.feature_engine.create_features(df_symbol)
            self.orchestrator.hmm_detector.train(df_features, self.orchestrator.feature_engine.feature_cols)
            results = self.orchestrator._run_enhanced_simulation(df_features, df_bench)
            
            equity_curve = [{"date": str(d).split(' ')[0], "value": float(v)} for d, v in results['equity'].items()]
            stats = {k: (0 if isinstance(v, float) and np.isnan(v) else v) for k, v in results['stats'].items()}
            return {"stats": stats, "equity_curve": equity_curve}
        except Exception as e: return {"error": str(e)}

    def run_diagnostics(self):
        checks = []
        try:
            d = self.orchestrator.data_engine.get_data("SPY", 10)
            checks.append({"name": "Data Feed", "status": "PASS", "details": f"{len(d)} bars"})
        except Exception as e: checks.append({"name": "Data Feed", "status": "FAIL", "details": str(e)})
        
        try:
            acct = self.orchestrator.execution_engine.get_account_info()
            checks.append({"name": "Broker Connection", "status": "PASS", "details": f"${acct.get('equity',0):,.2f}"})
        except Exception as e: checks.append({"name": "Broker Connection", "status": "FAIL", "details": str(e)})
        
        checks.append({"name": "AI Model", "status": "PASS" if self.orchestrator.hmm_detector.is_trained else "WARN", "details": "Ready"})
        return checks

    def _run_loop(self):
        print("--- AUTOMATION ENGAGED ---")
        self.orchestrator._run_startup_checks()
        
        # Safety Overrides
        CFG.MAX_POSITION_SIZE = 0.40
        CFG.MAX_GROSS_EXPOSURE = 0.95
        RISK_MANAGER.cfg = CFG
        
        while self.running:
            try:
                # 1. UPDATE METRICS & HISTORY
                price = self.orchestrator.execution_engine.get_price(CFG.SYMBOL)
                qty, val, _ = self.orchestrator.execution_engine.get_position(CFG.SYMBOL)
                cash_info = self.orchestrator.execution_engine.get_account_info()
                cash = float(cash_info.get('cash', 0) if cash_info else STATE.state['cash'])
                real_equity = cash + (qty * price)
                
                # Update last point OR add new point
                now_str = datetime.now().strftime("%H:%M")
                
                # Check if we should append or update
                if self.equity_history and self.equity_history[-1]['timestamp'] == now_str:
                    # Just update the last second to keep it live
                    self.equity_history[-1] = {"timestamp": now_str, "value": real_equity, "price": price}
                else:
                    self.equity_history.append({"timestamp": now_str, "value": real_equity, "price": price})
                
                if len(self.equity_history) > 200: self.equity_history.pop(0)
                self.save_history()
                
                STATE.update(equity=real_equity, cash=cash, position_qty=qty)

                # 2. ANALYSIS
                df = self.orchestrator.data_engine.get_data(CFG.SYMBOL, 365)
                df_bench = self.orchestrator.data_engine.get_data(CFG.BENCHMARK, 365)

                if not df.empty:
                    df_feats = self.orchestrator.feature_engine.create_features(df)
                    self.orchestrator.hmm_detector.train(df_feats, self.orchestrator.feature_engine.feature_cols)
                    regime = self.orchestrator.hmm_detector.predict(df_feats, self.orchestrator.feature_engine.feature_cols)
                    corr = self.orchestrator.risk_manager.calculate_correlation(df, df_bench)
                    signal = self.orchestrator.strategy.generate_signal(df_feats, regime, corr)
                    
                    atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
                    direction = 1 if signal['signal'] >= 0 else -1
                    
                    self.latest_signal = {
                        "sentiment": "BULLISH" if signal['signal'] > 0 else "BEARISH",
                        "strength": abs(signal['signal']),
                        "reason": f"Regime: {regime['regime']} | Corr: {corr:.2f}",
                        "atr": float(atr),
                        "targets": {
                            "entry": price,
                            "sl": price - (2.0 * atr * direction),
                            "tp": price + (4.0 * atr * direction)
                        }
                    }

                    # EXECUTION
