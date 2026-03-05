from app.routers import analysis
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.services.legal_engine import load_reasoning_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("System Startup: Pre-loading AI Models...")
    load_reasoning_model() 
    yield
    print("System Shutdown")

app = FastAPI(title="Virtual Senior Prosecutor API", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],  # Streamlit default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis.router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Server is running. Go to /docs to test the API."}