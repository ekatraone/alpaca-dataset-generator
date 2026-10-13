import json
import csv
import re
import torch
import os
from typing import List, Dict, Any, Optional
from docx import Document
import PyPDF2
import nltk
from nltk.corpus import stopwords
from collections import Counter
from transformers import PreTrainedTokenizer, PreTrainedModel
from sentence_transformers import SentenceTransformer

from config import CONFIG

_STOP_WORDS: Optional[set] = None

def _stop_words() -> set:
    """Load NLTK stopwords on first use so API-mode runs never need NLTK data."""
    global _STOP_WORDS
    if _STOP_WORDS is None:
        try:
            _STOP_WORDS = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            _STOP_WORDS = set(stopwords.words('english'))
    return _STOP_WORDS

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

def split_sentences(text: str) -> List[str]:
    """Whitespace-normalise and split on sentence-ending punctuation; no NLTK data needed."""
    sentences = []
    for para in text.split('\n\n'):
        para = ' '.join(para.split())
        if para:
            sentences.extend(s for s in re.split(r'(?<=[.!?])\s+', para) if s.strip())
    return sentences

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
    sentences = split_sentences(text)
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

def extract_keywords(text: str, n: Optional[int] = None) -> List[str]:
    """Extract the most common non-stopword keywords from the text."""
    n = n or CONFIG['keyword_count']
    stop_words = _stop_words()
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

# ---------------------------------------------------------------------------
# Output cleaning + validation
# ---------------------------------------------------------------------------

# Labels used in the few-shot prompts; models frequently echo them back.
_OUTPUT_LABEL_RE = re.compile(
    r'^\s*(?:summary|keywords?|title|sentiment|question|paraphrase|output|answer|'
    r'explanation|example|relation|learning path|assessment|difficulty|misconception|'
    r'analogy|application)\s*:\s*',
    re.IGNORECASE,
)

def clean_output(output: str) -> str:
    """Strip an echoed prompt label (e.g. 'Summary: ') and surrounding whitespace/quotes."""
    output = output.strip()
    output = _OUTPUT_LABEL_RE.sub('', output, count=1)
    # Drop anything after a model starts a new few-shot case
    for marker in ('\nText:', '\nConcept:', '\nNow ', '\nExample:'):
        idx = output.find(marker)
        if idx > 0:
            output = output[:idx]
    return output.strip().strip('"').strip()

def _word_count(text: str) -> int:
    return len(text.split())

def validation_failure(
    instruction_type: str,
    output: str,
    input_text: str,
    sentence_model: SentenceTransformer,
) -> Optional[str]:
    """
    Return a short reason the output is invalid, or None if it passes.
    Checks: generic length, semantic similarity to the input, then a per-type shape rule.
    """
    if not output:
        return "empty"
    words = _word_count(output)
    if words < 3 or len(output) < 10:
        return "too_short"

    input_embedding = sentence_model.encode(input_text, convert_to_tensor=True)
    output_embedding = sentence_model.encode(output, convert_to_tensor=True)
    similarity = torch.cosine_similarity(input_embedding, output_embedding, dim=0).item()
    if similarity < CONFIG['min_similarity_threshold']:
        return "low_similarity"

    lower = output.lower()

    # ---- standard generator types ----
    if instruction_type == "summarize":
        return None if 8 <= words <= 40 else "summary_length"
    if instruction_type == "keyword":
        return None if 3 <= len([k for k in output.split(',') if k.strip()]) <= 6 else "keyword_count"
    if instruction_type == "title":
        return None if 3 <= words <= 12 else "title_length"
    if instruction_type == "sentiment":
        return None if lower.startswith(('positive', 'negative', 'neutral')) else "no_sentiment_label"
    if instruction_type in ("question", "generate_question"):
        return None if output.rstrip().endswith('?') else "not_a_question"
    if instruction_type == "paraphrase":
        return None if words >= 8 and output.strip() != input_text.strip() else "paraphrase_unchanged_or_short"

    # ---- educational generator types ----
    if instruction_type == "concept_explanation":
        return None if words >= CONFIG['min_explanation_word_count'] else "explanation_too_short"
    if instruction_type in ("provide_example", "analogy", "application", "misconception", "concept_relation"):
        return None if words >= CONFIG['min_word_count'] else "too_few_words"
    if instruction_type == "learning_path":
        steps = re.findall(r'(?m)^\s*(?:\d+[.)]|[-*])\s+\S', output)
        ok = len(steps) >= CONFIG['min_learning_path_steps'] or output.count(',') >= CONFIG['min_learning_path_steps']
        return None if ok else "too_few_steps"
    if instruction_type == "quiz_generation":
        options = re.findall(r'(?mi)^\s*[A-D][.)]\s+\S', output)
        return None if len(options) >= CONFIG['min_quiz_options'] and '?' in output else "quiz_shape"
    if instruction_type == "difficulty_assessment":
        return None if any(k in lower for k in CONFIG['difficulty_keywords']) else "no_difficulty_label"

    return None


def is_valid_output(instruction_type: str, output: str, input_text: str, sentence_model: SentenceTransformer) -> bool:
    return validation_failure(instruction_type, output, input_text, sentence_model) is None

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

def from_sharegpt(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a ShareGPT record back to Alpaca format (inverse of to_sharegpt)."""
    turns = record.get("conversations", [])
    human = next((t["value"] for t in turns if t.get("from") == "human"), "")
    gpt = next((t["value"] for t in turns if t.get("from") == "gpt"), "")
    instruction, _, inp = human.partition("\n\n")
    return {
        "instruction": instruction,
        "input": inp,
        "output": gpt,
        "instruction_type": record.get("instruction_type", "unknown"),
    }

def load_existing_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load existing JSONL records (Alpaca or ShareGPT) as Alpaca dicts, for resume support."""
    if not os.path.exists(file_path):
        return []
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "conversations" in rec:
                rec = from_sharegpt(rec)
            if "output" in rec:
                records.append(rec)
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
    embeddings = torch.nn.functional.normalize(embeddings.float(), dim=1)

    # Greedy pass: keep an item unless it is too similar to something already kept.
    # Kept embeddings live in a preallocated matrix so each step is one matmul.
    kept = torch.empty_like(embeddings)
    kept_indices: List[int] = []
    for i in range(embeddings.shape[0]):
        emb = embeddings[i]
        if kept_indices:
            max_sim = (kept[:len(kept_indices)] @ emb).max().item()
            if max_sim >= similarity_threshold:
                continue
        kept[len(kept_indices)] = emb
        kept_indices.append(i)

    removed = len(examples) - len(kept_indices)
    print(f"Deduplication: removed {removed} near-duplicates, kept {len(kept_indices)}")
    return [examples[i] for i in kept_indices]
