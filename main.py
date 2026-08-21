import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load env variables
load_dotenv()

app = FastAPI(title="AI Fashion Stylist - ML Service")

# Setup CORS
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:5000")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routes
from routes.tryon import router as tryon_router

# Include routes
app.include_router(tryon_router, prefix="/tryon", tags=["tryon"])

@app.get("/")
def health_check():
    return {"status": "AI service running"}
