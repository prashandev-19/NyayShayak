from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.services import ocr_service, translation_service, rag_service, legal_engine
import uuid
import logging

# Setup Logger to track failures
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/analyze-case-rag")
async def analyze_case_file_rag(file: UploadFile = File(...)):
    try:
        # 0. Basic Validation
        if file.content_type not in ["application/pdf", "image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Only PDF or Images allowed.")

        # 1. OCR (Extract Hindi)
        file_bytes = await file.read()
        hindi_text = await ocr_service.extract_text_from_file(file_bytes, file.filename)
        
        if not hindi_text:
            raise HTTPException(status_code=400, detail="OCR failed to extract text.")

        # 2. Translation (Hindi -> English)
        english_text = await translation_service.translate_to_english(hindi_text)
        
        
        case_id = str(uuid.uuid4())
        await rag_service.store_case_data(english_text, case_id)
        
        analysis_result = await legal_engine.analyze_legal_case(case_id)
        
        return {
            "status": "success",
            "case_id": case_id,
            "original_snippet": hindi_text[:200], 
            "rag_analysis": analysis_result["analysis"] 
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error analyzing case: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"status": "error", "message": str(e)}
        )