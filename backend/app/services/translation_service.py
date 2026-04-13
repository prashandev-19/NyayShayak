from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import re
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

MODEL_NAME = "sarvamai/sarvam-translate"
USE_GPU = os.getenv("TRANSLATION_USE_GPU", "true").lower() == "true"
DEVICE = "cuda" if (torch.cuda.is_available() and USE_GPU) else "cpu"

# Tunable knobs for throughput + quality.
MAX_SOURCE_TOKENS_PER_CHUNK = int(os.getenv("TRANSLATION_MAX_SOURCE_TOKENS", "700"))
TRANSLATION_BATCH_SIZE = int(os.getenv("TRANSLATION_BATCH_SIZE", "4"))
CHUNK_OVERLAP_SENTENCES = int(os.getenv("TRANSLATION_CHUNK_OVERLAP_SENTENCES", "1"))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("TRANSLATION_MAX_NEW_TOKENS", "512"))
TOKENIZER_MAX_INPUT_TOKENS = int(os.getenv("TRANSLATION_MAX_INPUT_TOKENS", "4096"))

print(f"Translation service will use: {DEVICE.upper()}")
print(f"Translation model: {MODEL_NAME} (local cache only)")

tokenizer = None
model = None


BASE_TRANSLATION_SYSTEM_PROMPT = (
    "You are a professional Hindi-to-English legal translator. "
    "Translate the user's Hindi text to English accurately, preserving legal meaning. "
    "Preserve all names, dates, FIR numbers, section numbers (IPC/BNS), places, and numeric values exactly. "
    "Do not summarize or omit any information. "
    "Output only the English translation."
)

RETRY_TRANSLATION_SYSTEM_PROMPT = (
    "You are a strict Hindi-to-English legal translator. "
    "Translate every line fully and faithfully. "
    "Do not skip text. Do not summarize. "
    "Preserve all numbers, section references, names, dates, and identifiers exactly. "
    "Output only the English translation."
)

EN_TO_HI_TRANSLATION_SYSTEM_PROMPT = (
    "You are a professional English-to-Hindi legal translator. "
    "Translate the user's English legal text to Hindi accurately, preserving legal meaning. "
    "Preserve all names, dates, FIR numbers, section numbers (IPC/BNS), places, and numeric values exactly. "
    "Do not summarize or omit any information. "
    "Output only the Hindi translation."
)


def load_model():
    global tokenizer, model

    if tokenizer is None or model is None:
        print(f"Loading Translation Model ({MODEL_NAME}) on {DEVICE}...")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        ).to(DEVICE)

        model.eval()
        print("Translation model loaded successfully!")

    return tokenizer, model


def _safe_pad_token_id(tok) -> int:
    if tok.pad_token_id is not None:
        return tok.pad_token_id
    if tok.eos_token_id is not None:
        return tok.eos_token_id
    return 0


def _split_sentences(text: str) -> List[str]:
    pieces = re.split(r'([।.!?\n]+)', text)
    sentences: List[str] = []
    for i in range(0, len(pieces), 2):
        part = pieces[i].strip()
        delim = pieces[i + 1] if i + 1 < len(pieces) else ""
        if part:
            sentences.append((part + delim).strip())
    return sentences


def _split_long_sentence_by_words(tok, sentence: str, max_tokens: int) -> List[str]:
    words = sentence.split()
    if not words:
        return []

    out: List[str] = []
    current: List[str] = []

    for word in words:
        candidate = " ".join(current + [word]).strip()
        token_len = len(tok.encode(candidate, add_special_tokens=False))
        if token_len <= max_tokens:
            current.append(word)
        else:
            if current:
                out.append(" ".join(current).strip())
            current = [word]

    if current:
        out.append(" ".join(current).strip())

    return out


def _build_token_aware_chunks(tok, text: str, max_tokens: int, overlap_sentences: int) -> List[str]:
    text = text.strip()
    if not text:
        return []

    sentences = _split_sentences(text)
    if not sentences:
        sentences = [text]

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(tok.encode(sentence, add_special_tokens=False))

        if sentence_tokens > max_tokens:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
                current_tokens = 0

            long_parts = _split_long_sentence_by_words(tok, sentence, max_tokens)
            chunks.extend([p for p in long_parts if p])
            continue

        if current and current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current).strip())

            if overlap_sentences > 0:
                overlap = current[-overlap_sentences:]
                current = overlap[:]
                current_tokens = sum(len(tok.encode(s, add_special_tokens=False)) for s in current)
            else:
                current = []
                current_tokens = 0

        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def _extract_numeric_anchors(text: str) -> set:
    # Captures section-like and numeric anchors to detect loss during translation.
    return set(re.findall(r'\b\d+[A-Za-z]?\b', text or ""))


def _quality_ok(source: str, translated: str) -> bool:
    if not translated or not translated.strip():
        return False

    src_len = max(1, len(source.strip()))
    tgt_len = len(translated.strip())
    ratio = tgt_len / src_len

    # Very small/large length can indicate dropped content or garbage output.
    if ratio < 0.30 or ratio > 3.50:
        return False

    src_nums = _extract_numeric_anchors(source)
    if len(src_nums) >= 3:
        tgt_nums = _extract_numeric_anchors(translated)
        preserved = len(src_nums.intersection(tgt_nums)) / max(1, len(src_nums))
        if preserved < 0.50:
            return False

    return True


def _build_prompt(tok, source: str, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": source},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _translate_batch(tok, mdl, batch_chunks: List[str], system_prompt: str) -> List[str]:
    prompts = [_build_prompt(tok, chunk, system_prompt) for chunk in batch_chunks]

    max_length = TOKENIZER_MAX_INPUT_TOKENS
    if getattr(tok, "model_max_length", None) and tok.model_max_length < 100000:
        max_length = min(max_length, int(tok.model_max_length))

    encoded = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(DEVICE)

    pad_token_id = _safe_pad_token_id(tok)

    with torch.no_grad():
        generated = mdl.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

    results: List[str] = []
    input_lengths = encoded["attention_mask"].sum(dim=1).tolist()

    for idx, prompt_len in enumerate(input_lengths):
        new_tokens = generated[idx][int(prompt_len):]
        text = tok.decode(new_tokens, skip_special_tokens=True).strip()
        results.append(text)

    return results


def _translate_chunks_batched(tok, mdl, chunks: List[str], batch_size: int, system_prompt: str) -> List[str]:
    translated: List[str] = []
    i = 0

    while i < len(chunks):
        batch = chunks[i:i + batch_size]
        print(f"Translating batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size} ({len(batch)} chunk(s))...")

        try:
            translated_batch = _translate_batch(tok, mdl, batch, system_prompt)
        except RuntimeError as err:
            msg = str(err).lower()
            if "out of memory" in msg and len(batch) > 1 and DEVICE == "cuda":
                print(f"OOM on batch of size {len(batch)}. Falling back to single-chunk translation.")
                torch.cuda.empty_cache()
                translated_batch = []
                for one in batch:
                    translated_batch.extend(_translate_batch(tok, mdl, [one], system_prompt))
            else:
                raise

        translated.extend(translated_batch)
        i += batch_size

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return translated


def _remove_overlap_repetition(previous_text: str, current_text: str, max_overlap_words: int = 40) -> str:
    prev_words = previous_text.split()
    curr_words = current_text.split()
    if not prev_words or not curr_words:
        return current_text

    max_k = min(max_overlap_words, len(prev_words), len(curr_words))
    prev_lower = [w.lower() for w in prev_words]
    curr_lower = [w.lower() for w in curr_words]

    best_k = 0
    for k in range(max_k, 4, -1):
        if prev_lower[-k:] == curr_lower[:k]:
            best_k = k
            break

    if best_k > 0:
        return " ".join(curr_words[best_k:]).strip()
    return current_text


def _merge_translated_chunks(parts: List[str]) -> str:
    if not parts:
        return ""
    merged = parts[0].strip()
    for part in parts[1:]:
        clean_part = _remove_overlap_repetition(merged, part.strip())
        if clean_part:
            merged = (merged + " " + clean_part).strip()
    return merged

async def translate_to_english(hindi_text: str) -> str:
    try:
        print(f"\n{'='*60}")
        print(f"Starting Translation")
        print(f"Input length: {len(hindi_text)} characters")
        print(f"{'='*60}")

        tok, mdl = load_model()

        chunks = _build_token_aware_chunks(
            tok,
            hindi_text,
            max_tokens=MAX_SOURCE_TOKENS_PER_CHUNK,
            overlap_sentences=CHUNK_OVERLAP_SENTENCES,
        )
        print(f"Text split into {len(chunks)} chunk(s)")

        if not chunks:
            return ""

        print(f"Batch size: {TRANSLATION_BATCH_SIZE}, max source tokens/chunk: {MAX_SOURCE_TOKENS_PER_CHUNK}")

        english_parts = _translate_chunks_batched(
            tok,
            mdl,
            chunks,
            batch_size=max(1, TRANSLATION_BATCH_SIZE),
            system_prompt=BASE_TRANSLATION_SYSTEM_PROMPT,
        )

        # Retry only low-quality chunks once with a stricter prompt.
        retries = 0
        for i, (src, out) in enumerate(zip(chunks, english_parts)):
            if not _quality_ok(src, out):
                retries += 1
                print(f"Retrying low-quality chunk {i + 1}/{len(chunks)}")
                retried = _translate_batch(tok, mdl, [src], RETRY_TRANSLATION_SYSTEM_PROMPT)[0]
                english_parts[i] = retried if retried else out

        if retries:
            print(f"Retried {retries} chunk(s) due to quality checks")

        english_translation = _merge_translated_chunks(english_parts)

        print(f"\nTranslation complete: {len(english_translation)} characters")
        print(f"First 200 chars: {english_translation[:200]}...")
        print(f"{'='*60}\n")

        if DEVICE == "cuda":
            # One cache clear at the end is safer than clearing for each chunk.
            torch.cuda.empty_cache()
            print("GPU cache cleared after translation")

        return english_translation

    except Exception as e:
        print(f"Translation Error: {e}")
        import traceback
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f"Error in translation: {str(e)}"


async def translate_to_hindi(english_text: str) -> str:
    """Translate English legal text to Hindi using the same local model pipeline."""
    try:
        if not english_text or not english_text.strip():
            return ""

        tok, mdl = load_model()
        chunks = _build_token_aware_chunks(
            tok,
            english_text,
            max_tokens=MAX_SOURCE_TOKENS_PER_CHUNK,
            overlap_sentences=CHUNK_OVERLAP_SENTENCES,
        )

        if not chunks:
            return ""

        hindi_parts = _translate_chunks_batched(
            tok,
            mdl,
            chunks,
            batch_size=max(1, TRANSLATION_BATCH_SIZE),
            system_prompt=EN_TO_HI_TRANSLATION_SYSTEM_PROMPT,
        )

        hindi_translation = _merge_translated_chunks(hindi_parts)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        return hindi_translation
    except Exception as e:
        print(f"Hindi Translation Error: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f"Error in translation: {str(e)}"
