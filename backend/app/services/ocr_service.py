import easyocr
import numpy as np
from pdf2image import convert_from_bytes
from PIL import Image
import os
from pathlib import Path
from dotenv import load_dotenv
import torch

load_dotenv()
POPPLER_PATH = os.getenv("POPPLER_PATH", None)

model_dir = Path(__file__).parent.parent.parent / 'easyocr_models'
model_dir.mkdir(parents=True, exist_ok=True)

reader = easyocr.Reader(
    ['hi', 'en'], 
    gpu=torch.cuda.is_available(), 
    model_storage_directory=str(model_dir),
    user_network_directory=str(model_dir)
) 

async def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """
    Determines if the file is an image or PDF and extracts text accordingly.
    """
    extracted_text = ""
    if filename.lower().endswith('.pdf'):
        # Convert PDF bytes to a list of images (one per page)
        try:
            images = convert_from_bytes(file_bytes, poppler_path=POPPLER_PATH)
            
            for i, image in enumerate(images):
                # Convert PIL Image to numpy array (EasyOCR expects numpy)
                image_np = np.array(image)
                
              
                result = reader.readtext(image_np, detail=0, paragraph=True)
                
                extracted_text += f"\n--- Page {i+1} ---\n"
                extracted_text += " ".join(result)
                
        except Exception as e:
            return f"Error processing PDF: {str(e)}"

    else:
        try:
          
            result = reader.readtext(file_bytes, detail=0, paragraph=True)
            extracted_text = " ".join(result)
        except Exception as e:
            return f"Error processing Image: {str(e)}"

    return extracted_text