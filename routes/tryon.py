from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from services.tryon_pipeline import run_tryon_pipeline

router = APIRouter()

class TryOnRequest(BaseModel):
    personImage: str
    outfitImage: str
    pipeline: Optional[str] = "mask-free"  # "mask-free" or "masked"
    category: Optional[str] = None         # "upper", "lower", "overall" (optional override)

@router.post("/")
async def perform_tryon(request: TryOnRequest, pipeline_param: Optional[str] = Query(None, alias="pipeline")):
    try:
        # Determine pipeline type prioritizing query parameter, then request body, fallback to default
        pipeline_type = pipeline_param or request.pipeline or "mask-free"
        
        # Execute the pipeline orchestrator
        result_path = run_tryon_pipeline(
            person_image_url=request.personImage,
            outfit_image_url=request.outfitImage,
            pipeline_type=pipeline_type,
            category=request.category
        )
        
        return {
            "status": "success",
            "result_image_url": result_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
