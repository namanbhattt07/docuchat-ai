from fastapi import APIRouter

router = APIRouter(tags=["system"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Return a lightweight liveness signal; no external services are checked yet."""
    return {"status": "ok", "service": "docuchat-api"}
