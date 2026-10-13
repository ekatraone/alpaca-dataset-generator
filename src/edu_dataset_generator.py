import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from utils import (
    preprocess_text, generate_gpt2_output, generate_t5_output, extract_keywords,
    is_valid_output, clean_output, split_sentences,
)
from config import CONFIG

MAX_ATTEMPTS = 5


def generate_dataset(
    num_examples: int,
    input_texts: List[str],
    models: Dict,
    backend=None,
) -> List[Dict[str, Any]]:
    """
    Generate `num_examples` educational records, balanced across instruction types.
    Each candidate is validated before being kept; up to MAX_ATTEMPTS per slot.
    """
    instruction_types: List[Tuple[str, str]] = CONFIG['instruction_types']
    jobs = [instruction_types[i % len(instruction_types)] for i in range(num_examples)]
    random.shuffle(jobs)

    def run(job):
        instruction_type, instruction = job
        for _ in range(MAX_ATTEMPTS):
            text_sample = _sample_concept(input_texts)
            output = clean_output(_generate_output(models, backend, instruction_type, instruction, text_sample))
            if is_valid_output(instruction_type, output, text_sample, models["sentence_model"]):
                return {
                    "instruction": instruction,
                    "input": text_sample,
                    "output": output,
                    "instruction_type": instruction_type,
                }
        return None

    results: List[Optional[Dict[str, Any]]] = []
    with tqdm(total=num_examples, desc="Generating examples", unit="example") as pbar:
        if backend is not None and backend.is_api():
            with ThreadPoolExecutor(max_workers=CONFIG.get('api_concurrency', 8)) as pool:
                for rec in pool.map(run, jobs):
                    results.append(rec)
                    pbar.update(1)
        else:
            for job in jobs:
                results.append(run(job))
                pbar.update(1)

    dataset = [r for r in results if r]
    dropped = len(results) - len(dataset)
    if dropped:
        print(f"{dropped} slots produced no valid output after {MAX_ATTEMPTS} attempts")
    return dataset


def _sample_concept(input_texts: List[str]) -> str:
    """Pick a random chunk and take a 2-4 sentence window as the 'concept'."""
    sentences = split_sentences(random.choice(input_texts)) or [random.choice(input_texts)]
    num_sentences = min(len(sentences), random.randint(2, 4))
    return preprocess_text(" ".join(sentences[:num_sentences]))


def _generate_output(models, backend, instruction_type: str, instruction: str, input_text: str) -> str:
    """Route generation to the appropriate backend / model."""

    # ---- API backend: single unified call ----
    if backend is not None and backend.is_api():
        prompt = f"{instruction}\nConcept: {input_text}\nAnswer:"
        return backend.generate(prompt, max_tokens=CONFIG.get('api_max_tokens', 200))

    # ---- Local backend: model-specific routing ----
    device = CONFIG['device']
    gpt2 = (models["gpt2_tokenizer"], models["gpt2_model"])
    t5 = (models["t5_tokenizer"], models["t5_model"])

    if instruction_type == "learning_path":
        return generate_t5_output(*t5, "summarize", f"{instruction}\nConcept: {input_text}\nLearning path:", device, max_length=150)
    if instruction_type == "difficulty_assessment":
        return generate_t5_output(*t5, "summarize", f"{instruction}\nConcept: {input_text}\nAssessment:", device, max_length=100)
    if instruction_type == "concept_relation":
        related = (extract_keywords(input_text, n=1) or ["this topic"])[0]
        prompt = f"{instruction}\nConcept 1: {input_text}\nConcept 2: {related}\nRelation:"
        return generate_gpt2_output(*gpt2, prompt, device, max_length=100)
    if instruction_type == "quiz_generation":
        prompt = f"{instruction}\nConcept: {input_text}\nQuestion:"
        return generate_gpt2_output(*gpt2, prompt, device, max_length=200)

    # concept_explanation, generate_question, provide_example, misconception, analogy, application
    prompt = f"{instruction}\nConcept: {input_text}\nAnswer:"
    return generate_gpt2_output(*gpt2, prompt, device, max_length=100)
