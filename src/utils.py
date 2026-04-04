import json
import csv
import re
import torch
import os
from typing import List, Dict, Any
from docx import Document
import PyPDF2
import nltk
from nltk.corpus import stopwords
from collections import Counter
from transformers import PreTrainedTokenizer, PreTrainedModel
from sentence_transformers import SentenceTransformer

nltk.download('stopwords', quiet=True)

# Import CONFIG if it's defined in a separate file
from config import CONFIG

# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

def read_text_file(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except IOError as e:
        print(f"Error reading text file {file_path}: {e}")
        return ""

def read_pdf_file(file_path: str) -> str:
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return '\n\n'.join([page.extract_text().strip() for page in reader.pages])
    except Exception as e:
        print(f"Error reading PDF file {file_path}: {e}")
        return ""

def read_docx_file(file_path: str) -> str:
    try:
        doc = Document(file_path)
        return '\n\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        print(f"Error reading DOCX file {file_path}: {e}")
        return ""

def read_markdown_file(file_path: str) -> str:
    """Read a Markdown file, stripping code fences and header markers."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        # Remove fenced code blocks entirely (they're not useful training prose)
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)
        # Strip ATX heading markers (# ## ###) but keep the heading text
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        return text
    except IOError as e:
        print(f"Error reading Markdown file {file_path}: {e}")
        return ""

def read_csv_file(file_path: str) -> str:
    """Read a CSV file and return each row as a newline-separated prose chunk."""
    try:
        chunks = []
        with open(file_path, 'r', encoding='utf-8', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Represent each row as "field: value" pairs joined by semicolons
                chunk = '; '.join(f"{k}: {v}" for k, v in row.items() if v and v.strip())
                if chunk:
                    chunks.append(chunk)
        return '\n\n'.join(chunks)
    except Exception as e:
        print(f"Error reading CSV file {file_path}: {e}")
        return ""

def read_json_file(file_path: str) -> str:
    """Read a JSON or JSONL file and return each record as a prose chunk."""
    try:
        chunks = []
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()

        # Try JSONL first (one JSON object per line)
        if content.startswith('{'):
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    chunks.append(_flatten_json(obj))
                except json.JSONDecodeError:
                    pass
        else:
            # Regular JSON — could be a list or a single object
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    chunks.append(_flatten_json(item))
            else:
                chunks.append(_flatten_json(data))

        return '\n\n'.join(chunks)
    except Exception as e:
        print(f"Error reading JSON file {file_path}: {e}")
        return ""

def _flatten_json(obj, prefix: str = '') -> str:
    """Recursively flatten a JSON object into 'key: value' prose."""
    parts = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            parts.append(_flatten_json(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            parts.append(_flatten_json(v, f"{prefix}[{i}]"))
    else:
        if prefix:
            parts.append(f"{prefix}: {obj}")
        else:
            parts.append(str(obj))
    return ' | '.join(p for p in parts if p)

def read_file(file_path: str) -> str:
    """Read content from a file based on its extension."""
    _, ext = os.path.splitext(file_path.lower())
    readers = {
        '.txt':  read_text_file,
        '.pdf':  read_pdf_file,
        '.docx': read_docx_file,
        '.md':   read_markdown_file,
        '.csv':  read_csv_file,
        '.json': read_json_file,
        '.jsonl': read_json_file,
    }
    if ext not in readers:
        raise ValueError(f"Unsupported file type: {ext}")
    return readers[ext](file_path)

# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks on sentence boundaries.

    Instead of hard-truncating at max_chars, this produces multiple chunks
    from long documents so no content is discarded.

    Args:
        text: Source text.
        chunk_size: Target maximum characters per chunk.
        overlap: Number of characters to repeat at the start of each new chunk
                 to preserve context across chunk boundaries.

    Returns:
        List of cleaned text chunks, each within chunk_size characters.
    """
    # Normalise whitespace
    paragraphs = [' '.join(p.split()) for p in text.split('\n\n') if p.strip()]
    sentences = []
    for para in paragraphs:
        # Split on sentence-ending punctuation followed by whitespace
        parts = re.split(r'(?<=[.!?])\s+', para)
        sentences.extend(p for p in parts if p.strip())

    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        s_len = len(sentence)
        if current_len + s_len > chunk_size and current:
            chunk = ' '.join(current)
            chunks.append(chunk)
            # Roll back by overlap characters worth of sentences
            rollback = []
            rollback_len = 0
            for sent in reversed(current):
                if rollback_len + len(sent) > overlap:
                    break
                rollback.insert(0, sent)
                rollback_len += len(sent)
            current = rollback
            current_len = rollback_len

        current.append(sentence)
        current_len += s_len

    if current:
        chunks.append(' '.join(current))

    return [c for c in chunks if len(c) >= 50]

def preprocess_text(text: str, max_chars: int = 2000) -> str:
    """
    Preprocess text for direct use as a single model input.
    For bulk document loading, prefer chunk_text() instead.
    """
    if len(text) > max_chars:
        text = text[:max_chars]
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return '\n\n'.join([' '.join(p.split()) for p in paragraphs])

def extract_keywords(text: str, n: int = 5) -> List[str]:
    """Extract the most common keywords from the text."""
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in text.split() if word.isalnum()]
    word_freq = Counter(word for word in words if word not in stop_words)
    return [word for word, _ in word_freq.most_common(n)]

def generate_gpt2_output(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    prompt: str,
    device: torch.device,
    max_length: int = 50
) -> str:
    """Generate output using a GPT-2 model."""
    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )
    
    generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text.strip()

def generate_t5_output(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    prefix: str,
    input_text: str,
    device: torch.device,
    max_length: int = 100
) -> str:
    """Generate output using a T5 model."""
    input_ids = tokenizer(f"{prefix}: {input_text}", return_tensors="pt", max_length=512, truncation=True).input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=max_length, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def is_valid_output(
    instruction_type: str,
    output: str,
    input_text: str,
    sentence_model: SentenceTransformer
) -> bool:
    """Validate the generated output based on instruction type and similarity to input."""
    if len(output.split()) < 3 or len(output) < 10:
        return False

    input_embedding = sentence_model.encode(input_text, convert_to_tensor=True)
    output_embedding = sentence_model.encode(output, convert_to_tensor=True)
    similarity = torch.cosine_similarity(input_embedding, output_embedding, dim=0).item()

    if similarity < 0.3:
        return False

    if instruction_type == "summarize" and (len(output.split()) > 30 or len(output.split()) < 10):
        return False
    if instruction_type == "keyword" and not (3 <= len(output.split(',')) <= 5):
        return False
    if instruction_type == "title" and (len(output.split()) > 10 or len(output.split()) < 3):
        return False
    if instruction_type == "sentiment" and not any(word in output.lower() for word in ['positive', 'negative', 'neutral']):
        return False
    if instruction_type == "question" and not output.endswith('?'):
        return False

    return True

# ---------------------------------------------------------------------------
# Output format conversion
# ---------------------------------------------------------------------------

def to_sharegpt(example: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an Alpaca-format example to ShareGPT multi-turn format."""
    human_turn = example["instruction"]
    if example.get("input", "").strip():
        human_turn = f"{example['instruction']}\n\n{example['input']}"
    return {
        "conversations": [
            {"from": "human", "value": human_turn},
            {"from": "gpt",   "value": example["output"]},
        ]
    }

def save_to_jsonl(data: List[Dict[str, Any]], output_file: str, fmt: str = "alpaca"):
    """
    Save data to a JSONL file.

    Args:
        data: List of Alpaca-format examples.
        output_file: Destination file path.
        fmt: 'alpaca' (default) or 'sharegpt'.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    converter = to_sharegpt if fmt == "sharegpt" else (lambda x: x)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(converter(item), f, ensure_ascii=False)
            f.write('\n')

def load_existing_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load existing JSONL records for resume support."""
    if not os.path.exists(file_path):
        return []
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records

def append_to_jsonl(item: Dict[str, Any], output_file: str, fmt: str = "alpaca"):
    """Append a single example to a JSONL file (used for streaming/resume)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    converter = to_sharegpt if fmt == "sharegpt" else (lambda x: x)
    with open(output_file, 'a', encoding='utf-8') as f:
        json.dump(converter(item), f, ensure_ascii=False)
        f.write('\n')

# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(
    examples: List[Dict[str, Any]],
    sentence_model: SentenceTransformer,
    similarity_threshold: float = 0.95,
) -> List[Dict[str, Any]]:
    """
    Remove near-duplicate examples using output embedding cosine similarity.

    An example is dropped if its output is more than `similarity_threshold`
    similar to any already-kept example's output.

    Args:
        examples: List of Alpaca-format dicts.
        sentence_model: SentenceTransformer for encoding outputs.
        similarity_threshold: Cosine similarity above which an example is a duplicate.

    Returns:
        Deduplicated list.
    """
    if not examples:
        return []

    outputs = [ex["output"] for ex in examples]
    embeddings = sentence_model.encode(outputs, convert_to_tensor=True, show_progress_bar=True)

    kept_indices = []
    kept_embeddings = []

    for i, emb in enumerate(embeddings):
        if not kept_embeddings:
            kept_indices.append(i)
            kept_embeddings.append(emb)
            continue

        kept_tensor = torch.stack(kept_embeddings)
        sims = torch.cosine_similarity(emb.unsqueeze(0), kept_tensor, dim=1)
        if sims.max().item() < similarity_threshold:
            kept_indices.append(i)
            kept_embeddings.append(emb)

    removed = len(examples) - len(kept_indices)
    print(f"Deduplication: removed {removed} near-duplicates, kept {len(kept_indices)}")
    return [examples[i] for i in kept_indices]
