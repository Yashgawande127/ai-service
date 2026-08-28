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
        pipeline_result = run_tryon_pipeline(
            person_image_url=request.personImage,
            outfit_image_url=request.outfitImage,
            pipeline_type=pipeline_type,
            category=request.category
        )
        
        response_data = {
            "status": "success",
            "result_image_url": pipeline_result["result_image_url"],
            "local_path": pipeline_result["local_path"]
        }
        
        if pipeline_result.get("cloudinary_upload_error"):
            response_data["cloudinary_upload_error"] = pipeline_result["cloudinary_upload_error"]
            
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
