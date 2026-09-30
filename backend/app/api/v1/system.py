from fastapi import APIRouter

from app.services.ocr import ocr_status
from app.services.ollama import status
from app.services.providers import vision_capability

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/status")
async def system_status():
    # `ollama` is unchanged (the UI's readiness check keys off it). `ocr` and
    # `vision` report what this install can genuinely do: OCR only if the
    # local engine is installed, vision only if a vision model is configured
    # and present -- never assumed.
    return {"ollama": await status(), "ocr": ocr_status(), "vision": (await vision_capability()).to_dict()}
