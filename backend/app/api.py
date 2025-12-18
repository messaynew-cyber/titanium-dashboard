from fastapi import APIRouter, HTTPException
from typing import List
from .schemas import ForceTradeRequest, LogEntry
from .wrapper import titanium

router = APIRouter()

# This endpoint now returns the FULL data package (Equity, History, Trades)
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
    success, msg = titanium.force_trade(trade.symbol, trade.side, trade.qty)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "executed", "order_id": msg}

@router.get("/health")
async def health_check():
    return {"status": "online", "version": "2.0.0"}
