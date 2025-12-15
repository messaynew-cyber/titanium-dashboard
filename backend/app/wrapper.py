import sys
import threading
import time
import json
import requests
from pathlib import Path
import pandas as pd
from datetime import datetime

# ==========================================
# TELEGRAM CONFIGURATION
# ==========================================
TG_BOT_TOKEN = "8248228272:AAEbiHCplES8-ko8-hDln4-jqEjHoiwMTwo"
TG_CHANNEL_ID = "-1003253651772"

def send_telegram(message):
    try:
        url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TG_CHANNEL_ID, "text": message, "parse_mode": "HTML"}
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"[ERROR] Telegram failed: {e}")

# ==========================================
# SYSTEM SETUP
# ==========================================
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

try:
    # Import RISK_MANAGER directly to force update
    from algo.titanium_v1 import TitaniumOrchestrator, STATE, CFG, RISK_MANAGER
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import Algo. {e}")
    sys.exit(1)

class TitaniumService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TitaniumService, cls).__new__(cls)
            cls._instance.orchestrator = TitaniumOrchestrator()
            cls._instance.running = False
            cls._instance.thread = None
            cls._instance.last_rebalance = datetime.now()
        return cls._instance

    def start_engine(self):
        if self.running: return {"status": "Already running"}
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        send_telegram("🚀 <b>TITANIUM ENGINE STARTED</b>\nSystem is online.")
        return {"status": "Started"}

    def stop_engine(self):
        self.running = False
        send_telegram("🛑 <b>TITANIUM ENGINE STOPPED</b>")
        return {"status": "Stopping..."}

    def _run_loop(self):
        print("--- AUTOMATED TRADING ENGAGED ---")
        self.orchestrator._run_startup_checks()
        
        # 1. FORCE GLOBAL CONFIG UPDATE
        print("[INFO] Forcing Safety Configs...")
        CFG.MAX_POSITION_SIZE = 0.40
        CFG.MAX_GROSS_EXPOSURE = 0.95
        RISK_MANAGER.cfg.MAX_POSITION_SIZE = 0.40
        RISK_MANAGER.cfg.MAX_GROSS_EXPOSURE = 0.95
        
        while self.running:
            try:
                # Circuit Breaker
                can_trade, reason = self.orchestrator.circuit_breaker.can_trade()
                if not can_trade:
                    print(f"[WARNING] HALTED: {reason}")
                    time.sleep(60)
                    continue

                # Data
                print(f"[INFO] Fetching {CFG.SYMBOL} data...")
                df_symbol = self.orchestrator.data_engine.get_data(CFG.SYMBOL, 365)
                df_benchmark = self.orchestrator.data_engine.get_data(CFG.BENCHMARK, 365)

                if not df_symbol.empty:
                    # Analysis
                    df_features = self.orchestrator.feature_engine.create_features(df_symbol)
                    self.orchestrator.hmm_detector.train(df_features, self.orchestrator.feature_engine.feature_cols)
                    regime_info = self.orchestrator.hmm_detector.predict(df_features, self.orchestrator.feature_engine.feature_cols)
                    correlation = self.orchestrator.risk_manager.calculate_correlation(df_symbol, df_benchmark)
                    signal = self.orchestrator.strategy.generate_signal(df_features, regime_info, correlation)
                    
                    # 2. THE HARD CLAMP (Safety Intervention)
                    raw_pct = signal['position_pct']
                    clamped_pct = max(min(raw_pct, 0.40), -0.40)
                    
                    if clamped_pct != raw_pct:
                        print(f"[SAFETY CLAMP] Reduced size from {raw_pct:.2%} to {clamped_pct:.2%}")
                        signal['position_pct'] = clamped_pct

                    print(f"[INFO] Analysis: Signal={signal['signal']:.2f} | Size={signal['position_pct']:.2%}")

                    # Execution Logic
                    current_price = self.orchestrator.execution_engine.get_price(CFG.SYMBOL)
                    current_qty, _, _ = self.orchestrator.execution_engine.get_position(CFG.SYMBOL)
                    
                    target_value = STATE.state['equity'] * signal['position_pct']
                    target_qty = int(target_value / current_price) if current_price > 0 else 0
                    
                    delta_shares = target_qty - current_qty
                    delta_value = abs(delta_shares * current_price)

                    if delta_value > 200:
                        print(f"[TRADE DETECTED] Adjusting by {delta_shares} shares...")
                        
                        is_valid, msg = self.orchestrator.risk_manager.validate_order(
                            CFG.SYMBOL, delta_shares, current_price, 
                            {CFG.SYMBOL: {'value': current_qty * current_price}}, 
                            STATE.state['equity']
                        )

                        if is_valid:
                            side = "buy" if delta_shares > 0 else "sell"
                            success, order_id = self.orchestrator.execution_engine.submit_order(
                                CFG.SYMBOL, abs(delta_shares), side
                            )
                            if success:
                                STATE.update(position_qty=target_qty, last_trade_time=datetime.now().isoformat())
                                log_msg = f"✅ <b>EXECUTION</b>\n{side.upper()} {abs(delta_shares)} {CFG.SYMBOL} @ ${current_price:.2f}"
                                print(f"[SUCCESS] {log_msg}")
                                send_telegram(log_msg)
                        else:
                            print(f"[RISK REJECT] {msg}")
                            if "volume" not in msg:
                                send_telegram(f"🛡 <b>RISK REJECT</b>\n{msg}")

                    STATE.update(current_regime=regime_info.get('regime', 1))

                # Sleep (4 mins)
                for _ in range(CFG.POLL_INTERVAL):
                    if not self.running: break
                    time.sleep(1)

            except Exception as e:
                print(f"[ERROR] {e}")
                time.sleep(60)

    def get_state(self):
        STATE.load()
        regime_map = {0: "Bear", 1: "Neutral", 2: "Bull"}
        return {
            "equity": STATE.state.get('equity', 0.0),
            "cash": STATE.state.get('cash', 0.0),
            "daily_pnl": STATE.state.get('daily_pnl', 0.0),
            "total_pnl": STATE.state.get('total_pnl', 0.0),
            "position_qty": STATE.state.get('position_qty', 0),
            "current_drawdown": STATE.state.get('current_drawdown', 0.0),
            "regime": regime_map.get(STATE.state.get('current_regime', 1), "Unknown"),
            "is_active": self.running,
            "last_update": STATE.state.get('last_update', "")
        }

    def get_logs(self, limit: int = 50):
        log_dir = Path(CFG.LOG_PATH)
        files = list(log_dir.glob("*.jsonl"))
        if not files: return []
        latest = max(files, key=lambda f: f.stat().st_mtime)
        logs = []
        try:
            with open(latest, 'r') as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    try: logs.append(json.loads(line))
                    except: continue
        except: pass
        return logs

    def force_trade(self, symbol, side, qty):
        res = self.orchestrator.execution_engine.submit_order(symbol, qty, side)
        if res[0]: send_telegram(f"🕹 <b>MANUAL</b>\n{side.upper()} {qty} {symbol}")
        return res

titanium = TitaniumService()
