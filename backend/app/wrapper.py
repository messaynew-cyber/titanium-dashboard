import sys
import threading
import time
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add the parent directory to sys.path to import the algo
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Import the Algo Modules
try:
    from algo.titanium_v1 import TitaniumOrchestrator, STATE, CFG
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
        return {"status": "Started"}

    def stop_engine(self):
        self.running = False
        return {"status": "Stopping..."}

    def _run_loop(self):
        print("--- AUTOMATED TRADING ENGAGED ---")
        self.orchestrator._run_startup_checks()
        
        while self.running:
            try:
                # 1. Circuit Breakers
                can_trade, reason = self.orchestrator.circuit_breaker.can_trade()
                if not can_trade:
                    print(f"[WARNING] HALTED: {reason}")
                    time.sleep(60)
                    continue

                # 2. Get Data
                print(f"[INFO] Fetching {CFG.SYMBOL} data...")
                df_symbol = self.orchestrator.data_engine.get_data(CFG.SYMBOL, 365)
                df_benchmark = self.orchestrator.data_engine.get_data(CFG.BENCHMARK, 365)

                if not df_symbol.empty:
                    # 3. Analyze
                    df_features = self.orchestrator.feature_engine.create_features(df_symbol)
                    self.orchestrator.hmm_detector.train(df_features, self.orchestrator.feature_engine.feature_cols)
                    regime_info = self.orchestrator.hmm_detector.predict(df_features, self.orchestrator.feature_engine.feature_cols)
                    
                    correlation = self.orchestrator.risk_manager.calculate_correlation(df_symbol, df_benchmark)
                    signal = self.orchestrator.strategy.generate_signal(df_features, regime_info, correlation)
                    
                    print(f"[INFO] Analysis: Signal={signal['signal']:.2f} | Regime={regime_info['regime']}")

                    # 4. EXECUTION LOGIC (The Hands)
                    # Calculate how many shares we SHOULD have based on the signal
                    current_price = self.orchestrator.execution_engine.get_price(CFG.SYMBOL)
                    current_qty, _, _ = self.orchestrator.execution_engine.get_position(CFG.SYMBOL)
                    
                    # Target Value = Equity * Position % (e.g. $100k * 0.20 = $20k)
                    target_value = STATE.state['equity'] * signal['position_pct']
                    target_qty = int(target_value / current_price) if current_price > 0 else 0
                    
                    # Difference
                    delta_shares = target_qty - current_qty
                    delta_value = abs(delta_shares * current_price)

                    # 5. The "Smart Filter"
                    # Only trade if the difference is worth more than $200 (Prevents spamming tiny trades)
                    # AND if cooldown passed (1 minute)
                    if delta_value > 200:
                        print(f"[TRADE DETECTED] Adjusting position by {delta_shares} shares...")
                        
                        # Validate with Risk Manager
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
                                STATE.update(
                                    position_qty=target_qty, 
                                    last_trade_time=datetime.now().isoformat()
                                )
                                print(f"[SUCCESS] Executed {side.upper()} {abs(delta_shares)} shares")
                        else:
                            print(f"[RISK REJECT] {msg}")
                    else:
                        print(f"[HOLD] Position optimal (Delta: ${delta_value:.2f} < $200)")

                    # Update UI
                    STATE.update(current_regime=regime_info.get('regime', 1))

                # Sleep for POLL_INTERVAL (Defined in config as 240s)
                for _ in range(CFG.POLL_INTERVAL):
                    if not self.running: break
                    time.sleep(1)

            except Exception as e:
                print(f"[ERROR] Logic Failure: {e}")
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
        return self.orchestrator.execution_engine.submit_order(symbol, qty, side)

titanium = TitaniumService()
