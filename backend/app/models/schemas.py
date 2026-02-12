from pydantic import BaseModel
from typing import List, Optional

# This defines the structure of the "Legal Ingredient Audit" response
class CaseAnalysisResponse(BaseModel):
    original_text_hindi: str
    translated_text_english: str
    
    # The AI's findings
    summary: str
    missing_ingredients: List[str]  # e.g., ["Witness statement missing", "No seizure memo"]
    ipc_to_bns_mapping: str         # e.g., "IPC 379 -> BNS 303"
    
    # Final Recommendation in Hindi for the Police Officer
    final_response_hindi: str