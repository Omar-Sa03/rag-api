from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse


UI_DIR = Path(__file__).resolve().parent
UI_STATIC_DIR = UI_DIR / "static"

router = APIRouter()


@router.get("/")
async def ui_index():
    return FileResponse(UI_STATIC_DIR / "index.html")
