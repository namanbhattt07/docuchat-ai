from fastapi import APIRouter

from app.services.ollama import status

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/status")
async def system_status():
    return {"ollama": await status()}
