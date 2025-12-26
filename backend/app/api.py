from fastapi import APIRouter, HTTPException
from typing import List
from .schemas import ForceTradeRequest, LogEntry
from .wrapper import titanium

router = APIRouter()

@router.get("/status")
async def get_system_status():
    return await titanium.get_data()

@router.get("/logs", response_model=List[LogEntry])
async def get_system_logs(limit: int = 50):
    return titanium.get_logs(limit)

@router.post("/control/start")
async def start_trading():
    return await titanium.start_engine()

@router.post("/control/stop")
async def stop_trading():
    return await titanium.stop_engine()

@router.post("/trade/force")
async def execute_force_trade(trade: ForceTradeRequest):
    success, msg = await titanium.force_trade(trade.symbol, trade.side, trade.qty)
    if not success:
        raise HTTPException(status_code=400, detail=str(msg))
    return {"status": "executed", "order_id": msg}

@router.post("/tools/backtest")
async def run_backtest(days: int = 180):
    return await titanium.run_backtest(days)

@router.get("/tools/diagnostics")
async def run_diagnostics():
    return await titanium.run_diagnostics()

@router.post("/tools/scan")
async def scan_market():
    # v13.3 runs automatically, but we can trigger a log update
    return {"status": "Scan triggered in background"}

@router.get("/health")
async def health_check():
    return {"status": "online"}
