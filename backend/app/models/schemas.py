from pydantic import BaseModel
from typing import List, Optional


class CaseAnalysisResponse(BaseModel):
    case_id: str
    summary: str
    offenses: List[str]         
    missing_evidence: List[str] 
    recommendation: str        