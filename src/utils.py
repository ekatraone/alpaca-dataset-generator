import json
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

def read_text_file(file_path: str) -> str:
    """Read content from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except IOError as e:
        print(f"Error reading text file {file_path}: {e}")
        return ""

def read_pdf_file(file_path: str) -> str:
    """Read content from a PDF file."""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            # Join pages with double newline to preserve paragraph breaks
            return '\n\n'.join([page.extract_text().strip() for page in reader.pages])
    except Exception as e:
        print(f"Error reading PDF file {file_path}: {e}")
        return ""

def read_docx_file(file_path: str) -> str:
    """Read content from a DOCX file."""
    try:
        doc = Document(file_path)
        return ' '.join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        print(f"Error reading DOCX file {file_path}: {e}")
        return ""

def read_file(file_path: str) -> str:
    """Read content from a file based on its extension."""
    _, ext = os.path.splitext(file_path.lower())
    if (ext == '.txt') or (ext == '.md'):
        return read_text_file(file_path)
    elif ext == '.pdf':
        return read_pdf_file(file_path)
    elif ext == '.docx':
        return read_docx_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def preprocess_text(text: str, max_chars: int = 2000) -> str:
    """
    Preprocess the input text by:
    1. Removing extra whitespace while preserving paragraph breaks
    2. Truncating to prevent exceeding model token limits
    
    Args:
        text: The input text to preprocess
        max_chars: Maximum number of characters to keep (approximate token limit)
    
    Returns:
        Preprocessed and potentially truncated text
    """
    # Truncate the text if it's too long
    if len(text) > max_chars:
        text = text[:max_chars]
    
    # Split into paragraphs, clean each paragraph, then join with double newline
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

def save_to_jsonl(data: List[Dict[str, Any]], output_file: str):
    """Save data to a JSONL file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
