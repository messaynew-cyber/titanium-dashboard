from fastapi import APIRouter, HTTPException
from .schemas import SystemState, LogEntry, ForceTradeRequest
from .wrapper import titanium
from typing import List

router = APIRouter()

@router.get("/status", response_model=SystemState)
async def get_system_status():
    return titanium.get_state()

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
    return {"status": "online", "version": "1.0.0"}
