import sys
import threading
import time
import json
import requests
from pathlib import Path
from datetime import datetime

# CONFIG
TG_BOT_TOKEN = "8248228272:AAEbiHCplES8-ko8-hDln4-jqEjHoiwMTwo"
TG_CHANNEL_ID = "-1003253651772"

# SETUP
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
            # HISTORY STORAGE
            cls._instance.equity_history = [] 
            cls._instance.trade_history = []
        return cls._instance

    def start_engine(self):
        if self.running: return {"status": "Already running"}
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        send_telegram("🚀 <b>TITANIUM STARTED</b>")
        return {"status": "Started"}

    def stop_engine(self):
        self.running = False
        send_telegram("🛑 <b>TITANIUM STOPPED</b>")
        return {"status": "Stopping..."}

    def _run_loop(self):
        print("--- AUTOMATION ENGAGED ---")
        self.orchestrator._run_startup_checks()
        
        # SAFETY OVERRIDES
        CFG.MAX_POSITION_SIZE = 0.40
        CFG.MAX_GROSS_EXPOSURE = 0.95
        RISK_MANAGER.cfg = CFG
        
        while self.running:
            try:
                # 1. RECORD HISTORY
                current_equity = STATE.state.get('equity', 100000)
                self.equity_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "value": current_equity
                })
                # Keep last 100 points
                if len(self.equity_history) > 100: self.equity_history.pop(0)

                # 2. CIRCUIT BREAKER
                can_trade, reason = self.orchestrator.circuit_breaker.can_trade()
                if not can_trade:
                    qty, _, _ = self.orchestrator.execution_engine.get_position(CFG.SYMBOL)
                    if qty != 0:
                        self.orchestrator.execution_engine.close_position(CFG.SYMBOL)
                        send_telegram(f"🚨 <b>LIQUIDATION</b>\nReason: {reason}")
                    time.sleep(300)
                    continue

                # 3. TRADING LOGIC
                df_symbol = self.orchestrator.data_engine.get_data(CFG.SYMBOL, 365)
                df_bench = self.orchestrator.data_engine.get_data(CFG.BENCHMARK, 365)

                if not df_symbol.empty:
                    df_feats = self.orchestrator.feature_engine.create_features(df_symbol)
                    self.orchestrator.hmm_detector.train(df_feats, self.orchestrator.feature_engine.feature_cols)
                    regime = self.orchestrator.hmm_detector.predict(df_feats, self.orchestrator.feature_engine.feature_cols)
                    corr = self.orchestrator.risk_manager.calculate_correlation(df_symbol, df_bench)
                    signal = self.orchestrator.strategy.generate_signal(df_feats, regime, corr)
                    
                    # CLAMP
                    clamped = max(min(signal['position_pct'], 0.40), -0.40)
                    signal['position_pct'] = clamped

                    # EXECUTE
                    price = self.orchestrator.execution_engine.get_price(CFG.SYMBOL)
                    curr_qty, _, _ = self.orchestrator.execution_engine.get_position(CFG.SYMBOL)
                    target_val = current_equity * signal['position_pct']
                    target_qty = int(target_val / price) if price > 0 else 0
                    delta = target_qty - curr_qty
                    
                    if abs(delta * price) > 200:
                        is_valid, msg = self.orchestrator.risk_manager.validate_order(
                            CFG.SYMBOL, delta, price, 
                            {CFG.SYMBOL: {'value': curr_qty * price}}, current_equity
                        )
                        if is_valid:
                            side = "buy" if delta > 0 else "sell"
                            success, oid = self.orchestrator.execution_engine.submit_order(CFG.SYMBOL, abs(delta), side)
                            if success:
                                STATE.update(position_qty=target_qty, last_trade_time=datetime.now().isoformat())
                                log = f"✅ <b>EXECUTION</b>\n{side.upper()} {abs(delta)} @ ${price:.2f}"
                                print(f"[SUCCESS] {log}")
                                send_telegram(log)
                                # Record Trade
                                self.trade_history.insert(0, {
                                    "time": datetime.now().strftime("%H:%M:%S"),
                                    "symbol": CFG.SYMBOL,
                                    "side": side.upper(),
                                    "qty": abs(delta),
                                    "price": price
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
            "history": self.equity_history,
            "trades": self.trade_history[:50]
        }
    
    def get_logs(self, limit=50):
        # ... (Keep existing log logic) ...
        try:
            files = list(Path(CFG.LOG_PATH).glob("*.jsonl"))
            if not files: return []
            with open(max(files, key=lambda f: f.stat().st_mtime), 'r') as f:
                return [json.loads(l) for l in f.readlines()[-limit:]]
        except: return []

    def force_trade(self, s, side, q):
        return self.orchestrator.execution_engine.submit_order(s, q, side)

titanium = TitaniumService()
