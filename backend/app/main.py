from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from .config import settings
from .api import router
import os

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix=settings.API_V1_STR)

# ROBUST STATIC FILE SERVING
static_dir = Path(__file__).resolve().parent.parent.parent / "static"
assets_dir = static_dir / "assets"

# 1. Mount assets if they exist (CSS/JS)
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

# 2. Serve React App (SPA Fallback)
@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    # Pass API requests through
    if full_path.startswith("api"):
        return {"error": "API route not found"}
        
    # Serve specific file if exists
    file_path = static_dir / full_path
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
        
    # Fallback to index.html
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
        
    return {"message": "Frontend not built or not found."}
