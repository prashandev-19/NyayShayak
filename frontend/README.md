# NyayShayak Frontend

A Streamlit-based web interface for the NyayShayak Virtual Senior Prosecutor system.

## Features

- 📄 **File Upload**: Support for PDF and image files (JPG, PNG)
- 🔄 **Real-time Processing**: Live progress tracking during analysis
- 📊 **Comprehensive Results**: Detailed case analysis with offenses, evidence gaps, and recommendations
- 💾 **Export Options**: Download results as JSON or formatted text reports
- ⚙️ **Configurable**: Easy backend API configuration
- 🎨 **User-Friendly UI**: Clean, intuitive interface with custom styling

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Running FastAPI backend server (on port 8000 by default)

### Installation

1. **Navigate to the frontend directory**:
   ```powershell
   cd frontend
   ```

2. **Create a virtual environment** (recommended):
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Configure the backend URL** (if different from localhost:8000):
   - Edit `.streamlit/secrets.toml`
   - Update `API_BASE_URL` to your backend URL

## Running the Application

### Option 1: Using Streamlit command

```powershell
streamlit run app.py
```

### Option 2: Custom port

```powershell
streamlit run app.py --server.port 8501
```

The application will open in your default browser at `http://localhost:8501`

## Usage

1. **Start the Backend**: Ensure your FastAPI backend is running
   ```powershell
   cd ..\backend
   uvicorn app.main:app --reload
   ```

2. **Start the Frontend**: In a separate terminal
   ```powershell
   cd frontend
   streamlit run app.py
   ```

3. **Upload a Document**: 
   - Click "Browse files" or drag and drop
   - Select a PDF or image file

4. **Analyze**: Click the "Analyze Case" button

5. **View Results**: Review the analysis results including:
   - Case summary
   - Identified offenses
   - Missing evidence
   - Recommendations

6. **Export**: Download results as JSON or text format

## Configuration

### Backend API URL

Edit `.streamlit/secrets.toml`:

```toml
API_BASE_URL = "http://localhost:8000"
```

For production deployment:
```toml
API_BASE_URL = "https://your-backend-domain.com"
```

### Streamlit Settings

Edit `.streamlit/config.toml` to customize:
- Theme colors
- Upload size limits
- Server settings

## Project Structure

```
frontend/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .streamlit/
│   ├── config.toml          # Streamlit configuration
│   └── secrets.toml         # API configuration (keep private)
└── README.md                # This file
```

## Troubleshooting

### Cannot connect to backend

- Verify backend server is running: `http://localhost:8000/docs`
- Check the API URL in `.streamlit/secrets.toml`
- Ensure no firewall blocking the connection

### File upload fails

- Check file format (PDF, JPG, JPEG, PNG only)
- Verify file size is under 200MB (configurable in config.toml)
- Check backend logs for specific errors

### Timeout errors

- Increase timeout in `app.py` (currently 300 seconds)
- Large files and complex documents take longer to process

## Development

### Adding new features

1. Edit `app.py`
2. The app auto-reloads on file changes
3. Use `st.write()` for debugging

### Custom styling

Modify the CSS in the `st.markdown()` section at the top of `app.py`

## Production Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Add secrets in dashboard settings
4. Deploy

### Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Security Notes

- Never commit `.streamlit/secrets.toml` to version control
- Use environment variables for sensitive data in production
- Enable CORS properly in production environments
- Use HTTPS for production deployments

## Support

For issues or questions:
- Check FastAPI backend logs
- Review Streamlit logs in terminal
- Ensure all dependencies are installed correctly

---

**NyayShayak** - AI-Powered Legal Case Analysis System
