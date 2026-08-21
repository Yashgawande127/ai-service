from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.tryon_pipeline import run_tryon_pipeline

router = APIRouter()

class TryOnRequest(BaseModel):
    personImage: str
    outfitImage: str

@router.post("/")
async def perform_tryon(request: TryOnRequest):
    try:
        # Stub response: returning the outfitImage passed in
        result_url = request.outfitImage
        
        # Real pipeline call placeholder for later:
        # result_url = run_tryon_pipeline(request.personImage, request.outfitImage)
        
        return {
            "status": "success",
            "result_image_url": result_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
