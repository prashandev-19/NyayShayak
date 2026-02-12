from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MODEL_NAME = "ai4bharat/indictrans2-indic-en-1B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN")

# Global variables
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    
    if tokenizer is None or model is None:
        print(f"Loading Translation Model on {DEVICE}...")
        
        # 1. Load Tokenizer (use_fast=False is CRITICAL)
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True, 
            token=HF_TOKEN,
            use_fast=False 
        )
        
        # 2. Load Model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True, 
            token=HF_TOKEN
        ).to(DEVICE)
        
        print("Translation model loaded successfully!")
        
    return tokenizer, model

async def translate_to_english(hindi_text: str) -> str:
    """
    Translates Hindi legal text to English using IndicTrans2.
    """
    try:
        # Load model
        tokenizer_instance, model_instance = load_model()
        
        src_lang = "hin_Deva"
        tgt_lang = "eng_Latn"
        
        labeled_text = f"{src_lang} {tgt_lang} {hindi_text}"

        # 3. Tokenize the LABELED text
        batch = tokenizer_instance(
            [labeled_text], 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(DEVICE)

        # 4. Generate
        generated_tokens = model_instance.generate(
            **batch, 
            forced_bos_token_id=tokenizer_instance.convert_tokens_to_ids(tgt_lang),
            use_cache=False, 
            min_length=0, 
            max_length=512, 
            num_beams=5, 
            num_return_sequences=1
        )

        # 5. Decode
        english_translation = tokenizer_instance.batch_decode(
            generated_tokens.detach().cpu().tolist(), 
            skip_special_tokens=True
        )[0]
        
        return english_translation

    except Exception as e:
        print(f"Translation Error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error in translation: {str(e)}"