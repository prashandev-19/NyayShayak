from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel 
import torch
import os
import json
from dotenv import load_dotenv
from app.services import rag_service
from fastapi.concurrency import run_in_threadpool

load_dotenv()


BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B" 
ADAPTER_ID = os.getenv("ADAPTER_PATH")
HF_TOKEN = os.getenv("HF_TOKEN")

tokenizer = None
model = None

def load_reasoning_model():
   
    global tokenizer, model
    
    if tokenizer is None or model is None:
        print(f"Loading Base Model: {BASE_MODEL_ID}...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Detected Hardware: {device.upper()}")

        try:
           
            tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_ID, 
                token=HF_TOKEN
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if device == "cuda":
                print("Activating 4-bit Quantization (bitsandbytes)...")
                from transformers import BitsAndBytesConfig
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_ID,
                    token=HF_TOKEN,
                    quantization_config=bnb_config,
                    device_map="auto"
                )
            else:
                print(" No GPU detected. Loading in Standard CPU Mode.")
                print(" Note: This will be slow and consume significant RAM.")
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_ID,
                    token=HF_TOKEN,
                    device_map="cpu",
                    torch_dtype=torch.float32 
                )

            # 3. Attach Adapter (Works on both CPU and GPU)
            print(f"Attaching Adapter: {ADAPTER_ID}...")
            model = PeftModel.from_pretrained(
                base_model,
                ADAPTER_ID,
                token=HF_TOKEN
            )
            
            print("FactLegalLlama Adapter Loaded Successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
            
    return tokenizer, model

async def analyze_legal_case(case_id: str):
    tokenizer_instance, model_instance = load_reasoning_model()
    
    # 1. Retrieve Context from RAG
    context_chunks = await rag_service.get_relevant_context(
        query="incident details, accused name, witnesses, evidence found, and specific allegations", 
        filter={"case_id": case_id}
    )
    
    if not context_chunks:
        return {"case_id": case_id, "error": "No relevant context found in case file."}

    formatted_context = "\n\n".join(context_chunks)

    # 2. Construct Prompt
    system_prompt = """
    You are a Senior Indian Public Prosecutor acting as a strict legal auditor. 
    Your job is to validate chargesheets before they are filed in court.
    
    CRITICAL INSTRUCTIONS:
    1. Base your answer ONLY on the provided Context.
    2. Output strictly in valid JSON format. 
    3. Do not provide conversational filler.
    """

    user_prompt = f"""
    Analyze the following extracted text from an FIR/Chargesheet.

    CASE FACTS (Retrieved Context):
    \"\"\"{formatted_context}\"\"\"

    TASKS:
    1. Identify the primary offenses (IPC/BNS sections).
    2. List missing "Ingredients of Offense" (crucial evidence gaps).
    3. Suggest immediate next steps for the Investigating Officer (IO).

    OUTPUT FORMAT (JSON):
    {{
      "summary": "Brief 2-line summary...",
      "offenses": ["Section 379 IPC", "Section 303(2) BNS"],
      "missing_evidence": ["No independent witness", "Seizure memo missing"],
      "recommendation": "Advice for the officer..."
    }}
    
    JSON RESPONSE:
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    input_ids = tokenizer_instance.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model_instance.device)

    # Handle context window limits
    if input_ids.shape[1] > 7000:
        print(f"Warning: Input too long ({input_ids.shape[1]} tokens). Truncating...")
        input_ids = input_ids[:, -7000:]

    # 4. Run Inference in Threadpool (to avoid blocking FastAPI)
    def run_inference():
        terminators = [
            tokenizer_instance.eos_token_id,
            tokenizer_instance.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        return model_instance.generate(
            input_ids,
            max_new_tokens=1024, 
            temperature=0.1,     
            top_p=0.9,
            do_sample=True,
            eos_token_id=terminators
        )

    outputs = await run_in_threadpool(run_inference)

    response_text = tokenizer_instance.decode(
        outputs[0][input_ids.shape[-1]:], 
        skip_special_tokens=True
    )

    # 5. Extract & Clean JSON
    try:
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        
        json_start = clean_text.find("{")
        json_end = clean_text.rfind("}") + 1
        
        if json_start != -1 and json_end != -1:
            json_str = clean_text[json_start:json_end]
            final_json = json.loads(json_str)
            return final_json
        else:
            return {"error": "No JSON found in response", "raw_text": response_text}
            
    except json.JSONDecodeError:
        return {"error": "JSON parsing failed", "raw_text": response_text}