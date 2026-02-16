from app.routers import analysis
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.services.legal_engine import load_reasoning_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    print("System Startup: Pre-loading AI Models...")
    load_reasoning_model() 
    yield
    print("System Shutdown")

app = FastAPI(title="Virtual Senior Prosecutor API", lifespan=lifespan)

app.include_router(analysis.router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Server is running. Go to /docs to test the API."}