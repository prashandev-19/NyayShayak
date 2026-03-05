# NyayShayak - Installation and Setup Guide

## Quick Start

### 1. Install Backend Dependencies

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```powershell
cd ..\frontend
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

### 3. Run the Application

**Terminal 1 - Backend**:
```powershell
cd backend
.\venv\Scripts\Activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend**:
```powershell
cd frontend
.\venv\Scripts\Activate
streamlit run app.py
```

### 4. Access the Application

- **Frontend UI**: http://localhost:8501
- **Backend API Docs**: http://localhost:8000/docs

## Project Structure

```
NyayShayak/
├── backend/                  # FastAPI Backend
│   ├── app/
│   │   ├── main.py          # FastAPI app entry point
│   │   ├── routers/         # API endpoints
│   │   ├── services/        # Business logic
│   │   └── models/          # Data schemas
│   ├── requirements.txt
│   └── README.md
│
├── frontend/                 # Streamlit Frontend
│   ├── app.py               # Streamlit app
│   ├── .streamlit/          # Configuration
│   ├── requirements.txt
│   └── README.md
│
└── SETUP.md                 # This file
```

## Detailed Setup Instructions

### Backend Setup

1. **Navigate to backend**:
   ```powershell
   cd backend
   ```

2. **Create virtual environment**:
   ```powershell
   python -m venv venv
   ```

3. **Activate virtual environment**:
   ```powershell
   .\venv\Scripts\Activate
   ```

4. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

5. **Download required models** (if needed):
   - EasyOCR models will be downloaded automatically on first run
   - Place any custom models in `easyocr_models/`

6. **Run the backend**:
   ```powershell
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup

1. **Navigate to frontend**:
   ```powershell
   cd frontend
   ```

2. **Create virtual environment**:
   ```powershell
   python -m venv venv
   ```

3. **Activate virtual environment**:
   ```powershell
   .\venv\Scripts\Activate
   ```

4. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

5. **Configure backend URL** (if different from localhost):
   - Edit `.streamlit/secrets.toml`
   - Update `API_BASE_URL`

6. **Run the frontend**:
   ```powershell
   streamlit run app.py
   ```

## Running Both Services

### Option 1: Separate Terminals (Recommended)

Open two PowerShell terminals:

**Terminal 1 (Backend)**:
```powershell
cd C:\Users\singh\Desktop\NyayShayak\backend
.\venv\Scripts\Activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 (Frontend)**:
```powershell
cd C:\Users\singh\Desktop\NyayShayak\frontend
.\venv\Scripts\Activate
streamlit run app.py
```

### Option 2: Using Start-Job (Background)

```powershell
# Start backend in background
Start-Job -ScriptBlock {
    cd C:\Users\singh\Desktop\NyayShayak\backend
    .\venv\Scripts\Activate
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
}

# Wait a bit for backend to start
Start-Sleep -Seconds 5

# Start frontend
cd frontend
.\venv\Scripts\Activate
streamlit run app.py
```

## Testing the Setup

### 1. Check Backend

Visit: http://localhost:8000/docs

You should see the FastAPI Swagger documentation.

### 2. Check Frontend

Visit: http://localhost:8501

You should see the NyayShayak web interface.

### 3. Test Connection

In the frontend:
1. Click "Check Server Status" in the sidebar
2. Should show "✅ Backend server is running"

### 4. Test Full Pipeline

1. Upload a sample PDF or image
2. Click "Analyze Case"
3. Wait for results
4. Download report

## Troubleshooting

### Backend Issues

**Port already in use**:
```powershell
# Use different port
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

**Module not found errors**:
```powershell
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**GPU/CUDA errors**:
```powershell
# Check GPU availability
python check_gpu.py
```

### Frontend Issues

**Cannot connect to backend**:
- Verify backend is running on port 8000
- Check `.streamlit/secrets.toml` has correct URL
- Disable firewall temporarily to test

**Port already in use**:
```powershell
# Use different port
streamlit run app.py --server.port 8502
```

**Import errors**:
```powershell
# Reinstall dependencies
pip install -r requirements.txt
```

### Common Issues

**CORS errors**:
- Backend has CORS enabled for localhost:8501
- If using different port, update backend/app/main.py

**Large file uploads failing**:
- Increase max upload size in frontend/.streamlit/config.toml
- Check backend timeout settings

**Slow processing**:
- First run downloads models (can take time)
- GPU acceleration recommended for large files
- OCR processing is intensive

## Environment Variables (Optional)

Create `.env` files for environment-specific configuration:

**backend/.env**:
```
ENVIRONMENT=development
LOG_LEVEL=INFO
MODEL_PATH=./easyocr_models
```

**frontend/.env**:
```
API_BASE_URL=http://localhost:8000
```

## Production Deployment

### Backend (FastAPI)

Use Gunicorn with Uvicorn workers:
```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend (Streamlit)

Deploy to Streamlit Cloud or use Docker:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY frontend/requirements.txt .
RUN pip install -r requirements.txt
COPY frontend/ .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## Security Considerations

- Never commit `.streamlit/secrets.toml`
- Use environment variables in production
- Enable HTTPS in production
- Implement authentication if needed
- Rate limit API endpoints
- Validate file uploads

## Performance Optimization

- Use GPU for faster OCR and AI processing
- Cache frequently used data
- Implement request queuing for large files
- Use CDN for static assets
- Monitor resource usage

## Support

- Check logs in both backend and frontend terminals
- Review error messages carefully
- Ensure all dependencies are installed
- Verify correct Python version (3.8+)

---

**NyayShayak** - Virtual Senior Prosecutor System
