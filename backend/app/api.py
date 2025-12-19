from fastapi import APIRouter, HTTPException
from typing import List
from .schemas import ForceTradeRequest, LogEntry
from .wrapper import titanium

router = APIRouter()

@router.get("/status")
async def get_system_status():
    return titanium.get_data()

@router.get("/logs", response_model=List[LogEntry])
async def get_system_logs(limit: int = 50):
    return titanium.get_logs(limit)

@router.post("/control/start")
async def start_trading():
    return titanium.start_engine()

@router.post("/control/stop")
async def stop_trading():
    return titanium.stop_engine()

@router.post("/trade/force")
async def execute_force_trade(trade: ForceTradeRequest):
    # wrapper.py now handles the cleanup internally
    success, msg = titanium.force_trade(trade.symbol, trade.side, trade.qty)
    if not success:
        # Pass the error message back to the frontend
        raise HTTPException(status_code=400, detail=str(msg))
    return {"status": "executed", "order_id": msg}

# --- MISSING ENDPOINTS RESTORED ---

@router.post("/tools/backtest")
async def run_backtest(days: int = 180):
    """Trigger a backtest on the server."""
    return titanium.run_backtest(days)

@router.get("/tools/diagnostics")
async def run_diagnostics():
    """Run system health checks."""
    return titanium.run_diagnostics()

@router.post("/tools/scan")
async def scan_market():
    """Force an immediate market analysis."""
    result = titanium.generate_signal_now()
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@router.get("/health")
async def health_check():
    return {"status": "online"}
