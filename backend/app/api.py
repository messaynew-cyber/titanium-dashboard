from fastapi import APIRouter, HTTPException
from typing import List
from .schemas import ForceTradeRequest, LogEntry
from .wrapper import titanium

router = APIRouter()

@router.get("/status")
async def get_system_status(): return titanium.get_data()

@router.get("/logs", response_model=List[LogEntry])
async def get_system_logs(limit: int = 50): return titanium.get_logs(limit)

@router.post("/control/start")
async def start(): return titanium.start_engine()

@router.post("/control/stop")
async def stop(): return titanium.stop_engine()

@router.post("/trade/force")
async def force(t: ForceTradeRequest):
    s, m = titanium.force_trade(t.symbol, t.side, t.qty)
    if not s: raise HTTPException(400, m)
    return {"status": "ok", "id": m}

@router.post("/tools/backtest")
async def backtest(days: int = 180): return titanium.run_backtest(days)

@router.get("/tools/diagnostics")
async def diag(): return titanium.run_diagnostics()

# NEW: SIGNAL BUTTON ENDPOINT
@router.post("/tools/scan")
async def scan_market():
    return titanium.generate_signal_now()

@router.get("/health")
async def health(): return {"status": "online"}
