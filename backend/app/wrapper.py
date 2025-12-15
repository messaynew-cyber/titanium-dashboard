import sys
import threading
import time
import json
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

try:
    from algo.titanium_v1 import TitaniumOrchestrator, STATE, CFG
except ImportError as e:
    print(f"ALGO IMPORT ERROR: {e}")
    # Mocking for build process if algo missing
    TitaniumOrchestrator = None

class TitaniumService:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TitaniumService, cls).__new__(cls)
            if TitaniumOrchestrator:
                cls._instance.orchestrator = TitaniumOrchestrator()
            cls._instance.running = False
            cls._instance.thread = None
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
        print("--- BACKGROUND ENGINE STARTED ---")
        if hasattr(self, 'orchestrator'):
            self.orchestrator._run_startup_checks()
        while self.running:
            try:
                # Simulating orchestration loop
                time.sleep(1) 
            except Exception as e:
                print(e)
                time.sleep(5)

    def get_state(self):
        if hasattr(self, 'orchestrator') and 'STATE' in globals():
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
        return {"equity": 0, "cash": 0, "daily_pnl": 0, "total_pnl": 0, "position_qty": 0, "current_drawdown": 0, "regime": "Offline", "is_active": False, "last_update": ""}

    def get_logs(self, limit: int = 50):
        if 'CFG' not in globals(): return []
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
        if hasattr(self, 'orchestrator'):
            return self.orchestrator.execution_engine.submit_order(symbol, qty, side)
        return False, "Orchestrator not loaded"

titanium = TitaniumService()
