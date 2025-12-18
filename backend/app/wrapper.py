import sys
import threading
import time
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

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
            # Load history if available, else init
            cls._instance.equity_history = [{"timestamp": datetime.now().strftime("%H:%M"), "value": CFG.INITIAL_CAPITAL}]
            cls._instance.trade_history = []
            cls._instance.latest_signal = {
                "sentiment": "WAITING", "reason": "Initializing...", "atr": 0,
                "targets": {"sl": 0, "tp": 0, "entry": 0}
            }
        return cls._instance

    def start_engine(self):
        if self.running: return {"status": "Already running"}
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        send_telegram("🚀 <b>TITANIUM ENGINE STARTED</b>")
        return {"status": "Started"}

    def stop_engine(self):
        self.running = False
        send_telegram("🛑 <b>TITANIUM STOPPED</b>")
        return {"status": "Stopping..."}

    # --- NEW: RESTORED BACKTEST FUNCTION ---
    def run_backtest(self, days=180):
        print(f"[BUSY] Running Backtest for {days} days...")
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
        except Exception as e:
            return {"error": str(e)}

    # --- NEW: RESTORED DIAGNOSTICS FUNCTION ---
    def run_diagnostics(self):
        checks = []
        # Data
        try:
            d = self.orchestrator.data_engine.get_data("SPY", 10)
            checks.append({"name": "Data Feed", "status": "PASS", "details": f"{len(d)} bars"})
        except Exception as e: checks.append({"name": "Data Feed", "status": "FAIL", "details": str(e)})
        # Broker
        try:
            acct = self.orchestrator.execution_engine.get_account_info()
            checks.append({"name": "Broker Connection", "status": "PASS", "details": f"${acct.get('equity',0):,.2f}"})
        except Exception as e: checks.append({"name": "Broker Connection", "status": "FAIL", "details": str(e)})
        # Model
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
                # 1. UPDATE METRICS (Always runs, ensuring data isn't static)
                price = self.orchestrator.execution_engine.get_price(CFG.SYMBOL)
                qty, val, _ = self.orchestrator.execution_engine.get_position(CFG.SYMBOL)
                cash_info = self.orchestrator.execution_engine.get_account_info()
                cash = float(cash_info.get('cash', 0) if cash_info else STATE.state['cash'])
                real_equity = cash + (qty * price)
                
                # Append to history
                self.equity_history.append({"timestamp": datetime.now().strftime("%H:%M"), "value": real_equity})
                if len(self.equity_history) > 100: self.equity_history.pop(0)
                
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
                    
                    # Generate Signal Data
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
                    clamped = max(min(signal['position_pct'], 0.40), -0.40)
                    signal['position_pct'] = clamped
                    target_val = real_equity * signal['position_pct']
                    target_qty = int(target_val / price) if price > 0 else 0
                    delta = target_qty - qty
                    
                    if abs(delta * price) > 200:
                        is_valid, msg = self.orchestrator.risk_manager.validate_order(
                            CFG.SYMBOL, delta, price, {CFG.SYMBOL: {'value': qty*price}}, real_equity
                        )
                        if is_valid:
                            side = "buy" if delta > 0 else "sell"
                            success, oid = self.orchestrator.execution_engine.submit_order(CFG.SYMBOL, abs(delta), side)
                            if success:
                                STATE.update(position_qty=target_qty)
                                log = f"✅ <b>EXECUTION</b>\n{side.upper()} {abs(delta)} @ ${price:.2f}"
                                print(f"[SUCCESS] {log}")
                                send_telegram(log)
                                self.trade_history.insert(0, {
                                    "time": datetime.now().strftime("%H:%M"),
                                    "symbol": CFG.SYMBOL, "side": side.upper(), "qty": abs(delta), "price": price
                                })

                    STATE.update(current_regime=regime.get('regime', 1))

                for _ in range(CFG.POLL_INTERVAL):
                    if not self.running: break
                    time.sleep(1)
            except Exception as e:
                print(f"[ERROR] {e}")
                time.sleep(60)

    def get_data(self):
        STATE.load()
        regime_map = {0: "Bear", 1: "Neutral", 2: "Bull"}
        return {
            "state": {
                "equity": STATE.state.get('equity', 0),
                "cash": STATE.state.get('cash', 0),
                "daily_pnl": STATE.state.get('daily_pnl', 0),
                "regime": regime_map.get(STATE.state.get('current_regime', 1), "Unknown"),
                "is_active": self.running,
                "drawdown": STATE.state.get('current_drawdown', 0)
            },
            "signal": self.latest_signal,
            "history": self.equity_history,
            "trades": self.trade_history
        }
    
    def get_logs(self, limit=50):
        try:
            files = list(Path(CFG.LOG_PATH).glob("*.jsonl"))
            if not files: return []
            with open(max(files, key=lambda f: f.stat().st_mtime), 'r') as f:
                return [json.loads(l) for l in f.readlines()[-limit:]]
        except: return []

    def force_trade(self, s, side, q):
        return self.orchestrator.execution_engine.submit_order(s, q, side)

titanium = TitaniumService()
