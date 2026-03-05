from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel 
import torch
import os
import json
import time
import re
from dotenv import load_dotenv
from app.services import rag_service
from fastapi.concurrency import run_in_threadpool

load_dotenv()

# Reduce GPU memory fragmentation
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B" 
ADAPTER_ID = os.getenv("ADAPTER_PATH")
HF_TOKEN = os.getenv("HF_TOKEN")

tokenizer = None
model = None

def load_reasoning_model():
   
    global tokenizer, model
    
    if tokenizer is None or model is None:
        print(f"Loading Base Model: {BASE_MODEL_ID}...")
        
        # Enhanced GPU detection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Detected Hardware: {device.upper()}")
        
        if device == "cuda":
            print(f"GPU Found: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"PyTorch: {torch.__version__}")
        else:
            print(f"PyTorch Version: {torch.__version__}")
            print(f"CUDA Built: {torch.version.cuda if torch.version.cuda else 'No'}")

        try:
           
            tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_ID, 
                token=HF_TOKEN
            )

            
            if tokenizer.pad_token is None:
              
                tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
            
            
            if tokenizer.chat_template is None:
                tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{% if system_message != '' %}<|start_header_id|>system<|end_header_id|>\n\n{{ system_message }}<|eot_id|>{% endif %}{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"

            if device == "cuda":
                print("Activating 4-bit Quantization (bitsandbytes) for GPU")
                from transformers import BitsAndBytesConfig
                
                # Calculate optimal GPU memory allocation
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"  Available VRAM: {gpu_memory:.2f} GB")
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_ID,
                    token=HF_TOKEN,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                print("Model loaded on GPU with 4-bit quantization")
            else:
                print("No GPU detected. Loading in Standard CPU Mode.")
                print("Note: This will be slow and consume significant RAM.")
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_ID,
                    token=HF_TOKEN,
                    device_map="cpu",
                    torch_dtype=torch.float32 
                )

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

def clean_ocr_text(text: str) -> str:
    """Fix common OCR artifacts before sending context to the LLM."""
   
    text = re.sub(r'-{2,}\s*Page\s*\d+\s*-{2,}', ' ', text, flags=re.IGNORECASE)
    
    text = re.sub(r'(?<=[A-Z]{2}) (?=[A-Z])', '', text)
    
    text = re.sub(r'\b(\d) (\d{2})\b', r'\1\2', text)
    text = re.sub(r'\b(\d) (\d) (\d)\b', r'\1\2\3', text)
   
    text = re.sub(r'\.{3,}', '.', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

async def analyze_legal_case(case_id: str):
   
    print(f"\n{'='*60}")
    print(f"Starting Legal Analysis for Case: {case_id}")
    print(f"{'='*60}\n")
    
   
    if torch.cuda.is_available():
       
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all operations to complete
        torch.cuda.empty_cache()
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"GPU Memory: {free_mem / 1024**3:.2f} GB free / {total_mem / 1024**3:.2f} GB total\n")
        

        if free_mem / 1024**3 < 2.0:
            print(f" WARNING: Low GPU memory ({free_mem / 1024**3:.2f} GB free)")
            print(f"Attempting aggressive cleanup...")
            import gc
            gc.collect()  # Release Python refs first
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_mem, total_mem = torch.cuda.mem_get_info()
            print(f"After cleanup: {free_mem / 1024**3:.2f} GB free\n")
    
    start_time = time.time()
    
    print("[1/4] Loading AI model...")
    model_start = time.time()
    tokenizer_instance, model_instance = load_reasoning_model()
    print(f"Model ready ({time.time() - model_start:.2f}s)\n")
    
    # 1. Retrieve Context from RAG - Get MORE chunks for comprehensive analysis
    print("[2/4] Retrieving case context from vector database...")
    rag_start = time.time()
    
    
    queries = [
        "complainant name accused name FIR details IPC sections charges offense",
        "incident description evidence witnesses crime details section act",
    ]
    
    all_chunks = []
    for query in queries:
        chunks = await rag_service.get_relevant_context(
            query=query, 
            filter={"case_id": case_id},
            top_k=3  
        )
        all_chunks.extend(chunks)
    
    # Remove duplicates while preserving order
    seen = set()
    context_chunks = []
    for chunk in all_chunks:
        if chunk not in seen:
            seen.add(chunk)
            context_chunks.append(chunk)
    
    print(f"Case context retrieved ({time.time() - rag_start:.2f}s)\n")
    
    if not context_chunks:
        return {"case_id": case_id, "error": "No relevant context found in case file."}

    full_context = "\n\n".join(context_chunks)
   
    full_context_cleaned = clean_ocr_text(full_context)
    
    formatted_context = full_context_cleaned[:4000]
    if len(full_context_cleaned) > 4000:
        print(f"Context truncated from {len(full_context_cleaned)} to 4000 chars to fit GPU memory")
    
    print("[2.1/4] Pre-extracting IPC/BNS sections from case document...")
    section_patterns = [
        r'(?:Section|Sec\.?|धारा)\s+([\d]+(?:\s*[/-]\s*[\d]+)*(?:\s*\([a-zA-Z0-9]+\))?)\s+(?:IPC|BNS|of\s+IPC|of\s+BNS)',
        r'(?:IPC|BNS)\s+(?:Section|Sec\.?|धारा)\s+([\d]+(?:\s*[/-]\s*[\d]+)*(?:\s*\([a-zA-Z0-9]+\))?)',
        r'(?:under|u/s|U/S)\s+(?:Section|Sec\.?)?\s*([\d]+(?:\s*[/-]\s*[\d]+)*(?:\s*\([a-zA-Z0-9]+\))?)\s+(?:IPC|BNS)',
        r'(\d{2,3}(?:\s*\([a-zA-Z0-9]+\))?)\s+(?:IPC|BNS)',
        r'धारा\s+([\d]+(?:\s*\([a-zA-Z0-9]+\))?)',
        r'(?:Sections?|Sec\.?)\s+([\d]+(?:\s*,\s*[\d]+)*(?:\s*,?\s*and\s+[\d]+)?)\s+(?:of\s+)?(?:IPC|BNS)',
        
        r'under\s+Sections?\s+([\d]+(?:\s*,\s*[\d]+)*(?:\s*,?\s*and\s+[\d]+)?)',
      
        r'Sections?\s+([\d]{3}(?:\s*,\s*[\d]{3})*)',
    ]
    
    pre_extracted_sections = set()
    for pattern in section_patterns:
        matches = re.findall(pattern, formatted_context, re.IGNORECASE)
        for match in matches:
           
            section_parts = re.split(r'[,\s]+(?:and\s+)?', match)
            for part in section_parts:
                section_num = re.sub(r'\s+', '', part.strip())
                if section_num and section_num.isdigit() or re.match(r'^\d+\([a-zA-Z0-9]+\)$', section_num):
                    pre_extracted_sections.add(section_num)
    
    pre_extracted_list = sorted(list(pre_extracted_sections), key=lambda x: int(re.match(r'\d+', x).group()))
    print(f"Pre-extracted {len(pre_extracted_list)} IPC/BNS sections: {pre_extracted_list}")
    
  
    print(f"\nCase Context summary:")
    print(f"  - Number of chunks: {len(context_chunks)}")
    print(f"  - Total context length: {len(formatted_context)} characters")
    print(f"  - Pre-extracted sections: {', '.join(pre_extracted_list[:10])}{'...' if len(pre_extracted_list) > 10 else ''}")
    print(f"  - First 300 chars of context:")
    print(f"    {formatted_context[:300]}...")
    
    
    debug_dir = "debug_analysis"
    os.makedirs(debug_dir, exist_ok=True)
    debug_file = os.path.join(debug_dir, f"case_{case_id[:8]}_context.txt")
    with open(debug_file, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(f"CASE CONTEXT FOR: {case_id}\n")
        f.write("="*80 + "\n\n")
        f.write("FULL CONTEXT BEING SENT TO LLM:\n")
        f.write("-"*80 + "\n")
        f.write(formatted_context)
        f.write("\n\n" + "="*80 + "\n")
        f.write(f"PRE-EXTRACTED SECTIONS: {pre_extracted_list}\n")
        f.write("="*80 + "\n")
    print(f"Saved context debug file: {debug_file}")
    print()
    
    # 2. Retrieve IPC/BNS Statute Context
    print("[2.5/4] Retrieving relevant IPC/BNS statutes from legal database")
    statute_start = time.time()
    
    # Create a query from the case context and pre-extracted sections to find relevant statutes
    sections_query_part = ' '.join([f"Section {s}" for s in pre_extracted_list[:15]])  # First 15 sections
    statute_query = f"{sections_query_part} IPC BNS sections theft robbery assault murder rape fraud cheating criminal breach kidnapping extortion wrongful restraint relevant to: {formatted_context[:500]}"
    
    try:
        statute_chunks = await rag_service.get_relevant_context(
            query=statute_query,
            collection_name="ipc_bns_statutes",
            filter={},  
            top_k=10 
        )
        
        if statute_chunks:
            raw_statutes = "\n\n".join(statute_chunks)
           
            formatted_statutes = raw_statutes[:1500]
            if len(raw_statutes) > 1500:
                print(f"Statutes truncated from {len(raw_statutes)} to 1500 chars")
            print(f"Retrieved {len(statute_chunks)} relevant statute sections ({time.time() - statute_start:.2f}s)")
            print(f"  - First 200 chars of statutes:")
            print(f"    {formatted_statutes[:200]}...")
        else:
            formatted_statutes = "Statute database not available. Analysis will be based on case document only."
            print(f" No statutes retrieved - database may not be ingested yet")
    except Exception as e:
        formatted_statutes = "Statute database not available. Analysis will be based on case document only."
        print(f"Error retrieving statutes: {str(e)}")
    
    print()

    # -----------------------------------------------------------------------
    # TRAINING-ALIGNED PROMPT  (FactLegalLlama fine-tuned on Supreme Court
    # judgment dataset with columns: Facts, Issue, Arguments of Petitioner,
    # Arguments of Respondent, Reasoning, Decision, Label)
    #
    # The model learned to generate "Reasoning:" and "Decision:" text when
    # given the other columns as context.  We map the FIR fields to that
    # column layout and let the model continue naturally.  We then parse the
    # generated Reasoning + Decision ourselves to build the structured JSON.
    # -----------------------------------------------------------------------

    sections_str = ', '.join(pre_extracted_list) if pre_extracted_list else "to be identified from document"

    _comp_hint = "the complainant"
    _acc_hint  = "the accused"
    for _line in formatted_context.split('\n'):
        if re.search(r'complainant', _line, re.IGNORECASE) and re.search(r'name\s*:', _line, re.IGNORECASE):
            _m = re.search(r'name\s*:\s*([A-Za-z ]{3,40})', _line, re.IGNORECASE)
            if _m:
                _comp_hint = _m.group(1).strip()
                break
    for _line in formatted_context.split('\n'):
        if re.search(r'accused', _line, re.IGNORECASE) and re.search(r'name\s*:', _line, re.IGNORECASE):
            _m = re.search(r'name\s*:\s*([A-Za-z @]{3,50})', _line, re.IGNORECASE)
            if _m:
                _acc_hint = _m.group(1).strip()
                break

    # One-shot example using the exact column names from the training dataset
    EXAMPLE_BLOCK = (
        "Facts: Complainant Meena Devi filed an FIR at Civil Lines Police Station against "
        "accused Ramesh Yadav alleging theft of gold ornaments valued at Rs. 50,000 on "
        "12.03.2025. The accused was seen near the complainant's house on the day of the "
        "incident. Case registered under Sections 379 and 34 IPC.\n\n"
        "Issue: Whether accused Ramesh Yadav is criminally liable under Sections 379 and 34 "
        "IPC for the alleged theft, and what additional evidence is required for successful "
        "prosecution.\n\n"
        "Arguments of Petitioner: The complainant submits that gold ornaments were stolen from "
        "her locked almirah. A neighbour witnessed the accused leaving the premises. The accused "
        "has no legitimate reason to be on the complainant's property.\n\n"
        "Arguments of Respondent: Investigation is at an early stage. The accused's exact role "
        "and the recovery of stolen articles are yet to be established.\n\n"
        "Reasoning: The FIR discloses a cognizable offence under Section 379 IPC (theft) and "
        "Section 34 IPC (common intention). The complainant has established a prima facie case "
        "through the neighbour's eyewitness account and the recovery of the stolen items. "
        "However, to sustain a chargesheet the prosecution must recover the stolen ornaments, "
        "record a formal statement from the eyewitness, and obtain forensic examination if "
        "fingerprints are available. A TIP (Test Identification Parade) should be conducted. "
        "Documentary proof of ownership of the stolen ornaments is essential.\n\n"
        "Decision: The Investigating Officer should file a chargesheet under Sections 379 and "
        "34 IPC after completing the investigation. Priority must be given to recovery of stolen "
        "property, eyewitness statements, and forensic evidence. A TIP must be conducted before "
        "the accused is produced before the Magistrate.\n\n"
    )

    raw_prompt = (
        f"{EXAMPLE_BLOCK}"
        f"Facts: {formatted_context[:2000]}\n\n"
        f"Issue: Whether {_acc_hint} is criminally liable under IPC/BNS Sections "
        f"{sections_str} as alleged in the FIR filed by {_comp_hint}, and what "
        f"evidence is required to establish guilt beyond reasonable doubt.\n\n"
        f"Arguments of Petitioner: {_comp_hint} alleges that {_acc_hint} committed "
        f"the offences described in the FIR. The sections invoked are {sections_str}.\n\n"
        f"Arguments of Respondent: The accused's exact role and motive are yet to be "
        f"fully established during investigation.\n\n"
        f"Reasoning:"
    )

    print("[3/4] Preparing input for AI model (training-data-aligned format)...")
    tokenize_start = time.time()

    encoded = tokenizer_instance(
        raw_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=3000,
        padding=False
    )
    input_ids = encoded.input_ids.to(model_instance.device)
    attention_mask = encoded.attention_mask.to(model_instance.device)

    print(f"Input tokenized: {input_ids.shape[1]} tokens ({time.time() - tokenize_start:.2f}s)\n")

    # 4. Run Inference in Threadpool (to avoid blocking FastAPI)
    print("[4/4] Running AI inference (this may take 30-120 seconds)...")
    print("Generating legal analysis...")
    inference_start = time.time()
    
    _eot_id = tokenizer_instance.convert_tokens_to_ids("<|eot_id|>")
    terminators = [tokenizer_instance.eos_token_id]
    if _eot_id is not None and _eot_id != tokenizer_instance.unk_token_id:
        terminators.append(_eot_id)
    
    def run_inference():
        with torch.no_grad(): 
            return model_instance.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=700,        
                do_sample=False,            
                pad_token_id=tokenizer_instance.pad_token_id,
                eos_token_id=terminators,
                repetition_penalty=1.05,    
                use_cache=True
            )

    try:
        outputs = await run_in_threadpool(run_inference)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\nGPU OOM Error during generation.")
            print(f"Freeing tensors and retrying with minimal context.\n")
            
            import gc
            saved_ids_cpu = input_ids[:, -2000:].cpu()  
            saved_mask_cpu = attention_mask[:, -2000:].cpu()
            del input_ids
            del attention_mask
            
           
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                free_mem = torch.cuda.mem_get_info()[0] / 1024**3
                print(f"After tensor delete + cleanup: {free_mem:.2f} GB free")
            
           
            input_ids_retry = saved_ids_cpu.to(model_instance.device)
            attention_mask_retry = saved_mask_cpu.to(model_instance.device)
            del saved_ids_cpu, saved_mask_cpu
            
            print(f"Retrying with {input_ids_retry.shape[1]} tokens, max_new_tokens=300...")
            outputs = await run_in_threadpool(lambda: model_instance.generate(
                input_ids_retry,
                attention_mask=attention_mask_retry,
                max_new_tokens=500,
                do_sample=False,
                pad_token_id=tokenizer_instance.pad_token_id,
                eos_token_id=terminators,
                repetition_penalty=1.05,
                use_cache=False
            ))
           
            input_ids = input_ids_retry
            attention_mask = attention_mask_retry
        else:
            raise e
    
    print(f"AI inference completed ({time.time() - inference_start:.2f}s)\n")

    response_text = tokenizer_instance.decode(
        outputs[0][input_ids.shape[-1]:], 
        skip_special_tokens=True
    )
    
   
    generated_tokens = len(outputs[0]) - input_ids.shape[-1]
    del outputs
    del input_ids
    del attention_mask
    
    if torch.cuda.is_available():
        import gc
        gc.collect() 
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"GPU cache cleared. Available memory: {free_mem:.2f} GB")
    
    print(f"Generated {generated_tokens} tokens")

    # 5. Parse model output (training-data Reasoning/Decision format)
    print("Parsing AI response...")
    print(f"Raw AI output (first 500 chars):\n{response_text[:500]}\n")

    
    debug_response_file = os.path.join(debug_dir, f"case_{case_id[:8]}_model_response.txt")
    with open(debug_response_file, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(f"MODEL RAW RESPONSE FOR: {case_id}\n")
        f.write("="*80 + "\n\n")
        f.write(response_text)
        f.write("\n\n" + "="*80 + "\n")
        f.write(f"TOKENS GENERATED: {generated_tokens}\n")
        f.write("="*80 + "\n")
    print(f"💾 Saved model response to: {debug_response_file}\n")

    # ------------------------------------------------------------------
    # 5. Parse model output — the model was trained on the SC judgment
    #    dataset, so it generates Reasoning: ... Decision: ... text.
    #    We extract those sections and build structured JSON from them.
    # ------------------------------------------------------------------

    def extract_section_text(text: str, header: str) -> str:
        
        pattern = rf'{re.escape(header)}\s*(.*?)(?=\n(?:Facts|Issue|Arguments of (?:Petitioner|Respondent)|Reasoning|Decision|Label)\s*:|$)'
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        
        idx = text.lower().find(header.lower())
        if idx != -1:
            remainder = text[idx + len(header):].lstrip(':').strip()
            return remainder[:1500]
        return ""

    def extract_names_from_context(context: str):
        complainant = "the complainant"
        accused = "the accused"

        doc_lines = context.split('\n')
        comp_section, acc_section = "", ""
        in_comp = in_acc = False

        for line in doc_lines:
            ll = line.lower()
            if 'complainant' in ll and ('detail' in ll or 'name' in ll):
                in_comp, in_acc = True, False
                comp_section += line + "\n"
            elif 'accused' in ll and ('detail' in ll or 'name' in ll):
                in_acc, in_comp = True, False
                acc_section += line + "\n"
            elif line.strip() and (line.isupper() or line.startswith('===')):
                in_comp = in_acc = False
            elif in_comp:
                comp_section += line + "\n"
                if len(comp_section) > 500:
                    in_comp = False
            elif in_acc:
                acc_section += line + "\n"
                if len(acc_section) > 500:
                    in_acc = False

        for pattern in [
            r'Name\s*[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
            r'(?:Shri|Smt\.?|Mr\.?|Mrs\.?|Ms\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
        ]:
            m = re.search(pattern, comp_section if comp_section else context[:1000], re.MULTILINE)
            if m and len(m.group(1).split()) >= 2:
                complainant = m.group(1).strip()
                break

        search_acc = acc_section if acc_section else context
        for pattern in [
            r'Name\s*[:\-]?\s*([^\n:]+?)(?:\s+S/o|\s+D/o|\s+W/o|\s+Father|\s+Address|Age:|\s*$)',
            r'(?:against|versus)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
        ]:
            m = re.search(pattern, search_acc, re.MULTILINE | re.IGNORECASE)
            if m:
                cand = re.sub(r'\s+', ' ', m.group(1).strip())
                cand = re.sub(r'\s*\([^)]*\)', '', cand)[:50]
                if cand and len(cand) > 2:
                    accused = "Unidentified accused" if any(w in cand.lower() for w in ['unknown', 'unidentified', 'not known']) else cand
                    break

        return complainant, accused

    def build_offenses_from_sections(sections: list, reasoning_text: str, context: str) -> list:
        KNOWN_SECTIONS = {
            "302": "Murder", "304": "Culpable homicide not amounting to murder",
            "304A": "Causing death by negligence", "307": "Attempt to murder",
            "323": "Voluntarily causing hurt", "324": "Hurt by dangerous weapons",
            "325": "Grievous hurt", "326": "Grievous hurt by dangerous weapons",
            "354": "Assault on woman with intent to outrage modesty",
            "376": "Rape", "379": "Theft", "380": "Theft in dwelling house",
            "382": "Theft after preparation for hurt", "392": "Robbery",
            "395": "Dacoity", "406": "Criminal breach of trust",
            "420": "Cheating and dishonestly inducing delivery of property",
            "447": "Criminal trespass", "448": "House-trespass",
            "454": "Lurking house-trespass", "457": "Lurking house-trespass by night",
            "34":  "Acts done by several persons in furtherance of common intention",
            "120B": "Criminal conspiracy", "149": "Unlawful assembly",
            "341": "Wrongful restraint", "342": "Wrongful confinement",
            "343": "Wrongful confinement for three or more days",
            "363": "Kidnapping", "364": "Kidnapping for ransom",
            "384": "Extortion", "386": "Extortion by putting a person in fear of death",
            "500": "Defamation", "506": "Criminal intimidation",
            "511": "Attempt to commit offence",
            "233": "Making or selling instruments for counterfeiting coin",
            "464": "Making a false document",
        }
        
        extra_sections = set(sections)
        for pattern in [
            r'Section\s+(\d{2,3}[A-Z]?(?:\([a-zA-Z0-9]+\))?)\s+(?:IPC|BNS)',
            r'(?:under|u/s|U/S)\s+(\d{2,3}[A-Z]?)\s+(?:IPC|BNS)',
        ]:
            for m in re.finditer(pattern, reasoning_text, re.IGNORECASE):
                extra_sections.add(m.group(1).strip())

        offenses = []
        for sec in sorted(extra_sections, key=lambda x: int(re.match(r'\d+', x).group()) if re.match(r'\d+', x) else 0):
            desc = KNOWN_SECTIONS.get(sec)
            if desc:
                offenses.append(f"Section {sec} IPC - {desc}")
            else:
                # Try to find it mentioned in the reasoning
                m = re.search(rf'Section\s+{re.escape(sec)}\s+(?:IPC/BNS|IPC|BNS)[^\n]{{0,80}}', reasoning_text, re.IGNORECASE)
                offenses.append(f"Section {sec} IPC/BNS" + (f" - {m.group(0).split('-', 1)[-1].strip()[:60]}" if m and '-' in m.group(0) else ""))

        if not offenses:
            # Keyword inference from context
            ctx_lower = context.lower()
            if 'theft' in ctx_lower or 'stolen' in ctx_lower:
                offenses.append("Section 379 IPC - Theft")
            if 'assault' in ctx_lower or 'hurt' in ctx_lower or 'beat' in ctx_lower:
                offenses.append("Section 323 IPC - Voluntarily causing hurt")
            if 'murder' in ctx_lower or 'killed' in ctx_lower:
                offenses.append("Section 302 IPC - Murder")
            if 'cheating' in ctx_lower or 'fraud' in ctx_lower:
                offenses.append("Section 420 IPC - Cheating")
            if not offenses:
                offenses.append("IPC/BNS sections to be determined after full investigation")

        return offenses

    def extract_missing_evidence_from_reasoning(reasoning: str, context: str) -> list:
        evidence = []
        evidence_keywords = r'(?:must|should|required?|需要|necessary|essential|obtain|collect|record|examine|recover|produce|establish|verify|need)'
        for sent in re.split(r'[.!?\n]', reasoning):
            if re.search(evidence_keywords, sent, re.IGNORECASE):
                sent = sent.strip()
                if 20 < len(sent) < 200:
                    evidence.append(sent)
        evidence = list(dict.fromkeys(evidence))[:5]  

       
        ctx_lower = context.lower()
        rule_based = []
        if 'theft' in ctx_lower or 'stolen' in ctx_lower or 'robbery' in ctx_lower:
            rule_based += ["Recovery and panchnama of stolen property", "Ownership proof of stolen items",
                           "CCTV footage from crime scene"]
        if 'assault' in ctx_lower or 'hurt' in ctx_lower or 'injury' in ctx_lower or 'beat' in ctx_lower:
            rule_based += ["Medical Examination Report (MLC)", "Photographs of injuries",
                           "Doctor's injury certificate"]
        if 'murder' in ctx_lower or 'death' in ctx_lower or 'killed' in ctx_lower:
            rule_based += ["Post-mortem report", "Forensic examination of scene", "Weapon recovery and FSL report"]
        if 'fraud' in ctx_lower or 'cheating' in ctx_lower or 'forgery' in ctx_lower:
            rule_based += ["Documentary evidence of transaction", "Bank statements or payment records",
                           "Forensic document examination"]

        rule_based += ["Independent witness statements", "Scene of crime inspection and site plan",
                       "Call Detail Records (CDR) if relevant"]

        for item in rule_based:
            if item not in evidence:
                evidence.append(item)

        return list(dict.fromkeys(evidence))[:8]

    def parse_judgment_output(response: str, context: str, sections: list) -> dict:
        """
        Parse model output that follows the SC judgment training format.
        Extract Reasoning: and Decision: columns, then build structured JSON.
        """
        print(f"Model output preview (first 400 chars): {response[:400]}")

        reasoning = extract_section_text(response, "Reasoning")
        decision  = extract_section_text(response, "Decision")

        if not reasoning and len(response.strip()) > 30:
            reasoning = response.strip()
        if not decision:
           
            parts = re.split(r'\nDecision\s*:', response, flags=re.IGNORECASE)
            decision = parts[1].strip()[:600] if len(parts) > 1 else ""

        print(f"Reasoning excerpt: {reasoning[:200]}")
        print(f"Decision  excerpt: {decision[:200]}")

        if not reasoning and not decision:
            print("Model produced no usable Reasoning/Decision — using fallback")
            return None

        complainant, accused = extract_names_from_context(context)
        offenses = build_offenses_from_sections(sections, reasoning, context)
        missing_evidence = extract_missing_evidence_from_reasoning(reasoning, context)

        offense_labels = ', '.join([o.split(' - ')[0] for o in offenses[:3]])
        summary = (
            f"Complainant {complainant} filed FIR against {accused} for "
            f"{', '.join([o.split(' - ')[-1] for o in offenses[:2]]).lower()} "
            f"under {offense_labels}. "
        )
        
        first_reasoning_sent = re.split(r'[.!?]', reasoning)[0].strip()
        if first_reasoning_sent and len(first_reasoning_sent) > 20:
            summary += first_reasoning_sent + "."

        
        if decision and len(decision.strip()) > 30:
            recommendation = decision.strip()[:600]
        else:
            recommendation = (
                f"The Investigating Officer should complete the investigation and file a chargesheet "
                f"under {offense_labels}. Priority: (1) Record statements of {complainant} and all witnesses; "
                f"(2) Conduct scene inspection and prepare panchnama; "
                f"(3) Collect all physical and documentary evidence; "
                f"(4) {'Identify accused through investigation' if 'unidentified' in accused.lower() else f'Arrest and question {accused}'}; "
                f"(5) Comply with all CrPC procedural requirements."
            )

        return {
            "summary": summary,
            "offenses": offenses,
            "missing_evidence": missing_evidence,
            "recommendation": recommendation,
        }

    def normalize_to_strings(data):
       
        if not isinstance(data, dict):
            return data
        normalized = {}
        for key, value in data.items():
            if key in ['offenses', 'missing_evidence'] and isinstance(value, list):
                normalized[key] = []
                for item in value:
                    if isinstance(item, dict):
                        if 'code' in item and 'description' in item:
                            normalized[key].append(f"Section {item['code']} - {item['description']}")
                        elif 'name' in item:
                            normalized[key].append(item['name'])
                        else:
                            normalized[key].append(str(item))
                    else:
                        normalized[key].append(str(item))
            else:
                normalized[key] = value
        return normalized

    try:
        final_json = parse_judgment_output(response_text, formatted_context, pre_extracted_list)

        if final_json:
            final_json = normalize_to_strings(final_json)

            required_fields = ['summary', 'offenses', 'missing_evidence', 'recommendation']
            for field in required_fields:
                if field not in final_json:
                    final_json[field] = [] if field in ['offenses', 'missing_evidence'] else "No data provided"

            total_time = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"✓ Legal Analysis Complete!")
            print(f"Total Time: {total_time:.2f}s")
            print(f"{'='*60}\n")

           
            if torch.cuda.is_available():
                import gc
                gc.collect() 
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"Final GPU cache cleanup complete")
            
            return final_json
        else:
            print("✗ Error: Could not parse response")
            if torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            return {"error": "Failed to parse response", "raw_text": response_text[:500]}
            
    except Exception as e:
        print(f"✗ Unexpected error during parsing: {str(e)}")
        if torch.cuda.is_available():
            import gc
            gc.collect()  
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return {"error": f"Parsing error: {str(e)}", "raw_text": response_text[:500] if 'response_text' in locals() else "No response generated"}