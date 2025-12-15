import sys
import threading
import time
import json
from pathlib import Path
import pandas as pd

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
        return cls._instance

    def start_engine(self):
        """Starts the live trading loop in a background thread."""
        if self.running:
            return {"status": "Already running"}
        
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        return {"status": "Started"}

    def stop_engine(self):
        """Stops the loop."""
        self.running = False
        return {"status": "Stopping..."}

    def _run_loop(self):
        """The threaded worker loop."""
        print("--- BACKGROUND TRADING ENGINE STARTED ---")
        
        # 1. Run Checks
        self.orchestrator._run_startup_checks()
        
        # 2. Run the REAL Live Loop
        # We override the infinite loop behavior to allow stopping
        while self.running:
            try:
                # This performs ONE iteration of the trading logic
                # We replicate the logic inside run_live but controlled
                
                # Check Circuit Breakers
                can_trade, reason = self.orchestrator.circuit_breaker.can_trade()
                if not can_trade:
                    print(f"[WARNING] Trading halted: {reason}")
                    time.sleep(60)
                    continue

                # Get Data
                print(f"[INFO] Fetching {CFG.SYMBOL} data...")
                df_symbol = self.orchestrator.data_engine.get_data(CFG.SYMBOL, 365)
                df_benchmark = self.orchestrator.data_engine.get_data(CFG.BENCHMARK, 365)

                if not df_symbol.empty:
                    # Features & AI
                    print("[INFO] Calculating features & HMM...")
                    df_features = self.orchestrator.feature_engine.create_features(df_symbol)
                    self.orchestrator.hmm_detector.train(df_features, self.orchestrator.feature_engine.feature_cols)
                    regime_info = self.orchestrator.hmm_detector.predict(df_features, self.orchestrator.feature_engine.feature_cols)
                    
                    # Strategy
                    correlation = self.orchestrator.risk_manager.calculate_correlation(df_symbol, df_benchmark)
                    signal = self.orchestrator.strategy.generate_signal(df_features, regime_info, correlation)
                    
                    print(f"[INFO] Signal: {signal['signal']:.2f} | Regime: {regime_info['regime']}")

                    # Update UI State
                    STATE.update(
                        current_regime=regime_info.get('regime', 1),
                        regime_confidence=regime_info.get('confidence', 0.0)
                    )
                    
                    # (Note: Actual trade execution logic would go here, matching run_live)
                
                # Sleep for 5 minutes (300s) or until stopped
                for _ in range(300):
                    if not self.running: break
                    time.sleep(1)

            except Exception as e:
                print(f"[ERROR] Loop Error: {e}")
                time.sleep(60)

    def get_state(self):
        """Reads the latest state."""
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
        """Reads the JSONL log file."""
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
