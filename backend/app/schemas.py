from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class SystemState(BaseModel):
    equity: float
    cash: float
    daily_pnl: float
    total_pnl: float
    position_qty: int
    current_drawdown: float
    regime: str
    is_active: bool
    last_update: str

class LogEntry(BaseModel):
    timestamp: str
    level: str
    message: str
    data: Optional[Dict[str, Any]] = None

class ForceTradeRequest(BaseModel):
    symbol: str = "GLD"
    side: str
    qty: int
