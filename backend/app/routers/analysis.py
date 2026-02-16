from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services import ocr_service, translation_service, rag_service, legal_engine
from app.models.schemas import CaseAnalysisResponse 
import uuid
import logging

# Setup Logger
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/analyze-case-rag", response_model=CaseAnalysisResponse)
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
        
        # 3. Store in RAG (Generate Case ID)
        case_id = str(uuid.uuid4())
        await rag_service.store_case_data(english_text, case_id)
        
        ai_result = await legal_engine.analyze_legal_case(case_id)
        
        if "error" in ai_result:
            logger.error(f"AI Error: {ai_result}")
            return CaseAnalysisResponse(
                case_id=case_id,
                summary="Error in AI Analysis",
                offenses=[],
                missing_evidence=[],
                recommendation=f"System Error: {ai_result.get('error')}"
            )

        return CaseAnalysisResponse(
            case_id=case_id,
            summary=ai_result.get("summary", "No summary provided."),
            offenses=ai_result.get("offenses", []),
            missing_evidence=ai_result.get("missing_evidence", []),
            recommendation=ai_result.get("recommendation", "No recommendation provided.")
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error analyzing case: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal Server Error: {str(e)}"
        )