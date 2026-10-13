import torch
from typing import List, Dict, Any, Optional
from collections import Counter
import random
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from utils import preprocess_text, generate_gpt2_output, generate_t5_output, extract_keywords, is_valid_output
from config import CONFIG


def generate_dataset(
    num_examples: int,
    input_texts: List[str],
    models: Dict,
    device: torch.device,
    backend=None,
) -> List[Dict[str, Any]]:
    dataset = []
    instruction_type_counts = Counter()
    INSTRUCTION_TYPES = CONFIG['instruction_types']

    with tqdm(total=num_examples, desc="Generating examples", unit="example") as pbar:
        while len(dataset) < num_examples:
            instruction_type, instruction = random.choice(INSTRUCTION_TYPES)

            if instruction_type_counts[instruction_type] >= num_examples // len(INSTRUCTION_TYPES):
                continue

            input_text = random.choice(input_texts)
            sentences = sent_tokenize(input_text)
            num_sentences = min(len(sentences), random.randint(2, 4))
            text_sample = " ".join(sentences[:num_sentences])
            text_sample = preprocess_text(text_sample)

            example = generate_example(models, device, instruction_type, instruction, text_sample, backend)
            if example:
                dataset.append(example)
                instruction_type_counts[instruction_type] += 1
                pbar.update(1)

    return dataset


def generate_example(
    models: Dict,
    device: torch.device,
    instruction_type: str,
    instruction: str,
    input_text: str,
    backend=None,
) -> Optional[Dict[str, Any]]:
    max_attempts = 5
    for _ in range(max_attempts):
        output = _generate_output(models, device, backend, instruction_type, instruction, input_text)

        if is_valid_output(instruction_type, output, input_text, models["sentence_model"]):
            return {
                "instruction": instruction,
                "input": input_text,
                "output": output,
            }
    return None


def _generate_output(
    models: Dict,
    device: torch.device,
    backend,
    instruction_type: str,
    instruction: str,
    input_text: str,
) -> str:
    """Route generation to the appropriate backend / model."""

    # ---- API backend: single unified call ----
    if backend is not None and backend.is_api():
        prompt = f"{instruction}\nConcept: {input_text}\nAnswer:"
        return backend.generate(prompt, max_tokens=CONFIG.get('api_max_tokens', 200))

    # ---- Local backend: model-specific routing ----
    if instruction_type == "learning_path":
        return generate_t5_output(
            models["t5_tokenizer"], models["t5_model"],
            "summarize", f"{instruction}\nConcept: {input_text}\nLearning path:",
            device, max_length=150,
        )
    if instruction_type == "difficulty_assessment":
        return generate_t5_output(
            models["t5_tokenizer"], models["t5_model"],
            "summarize", f"{instruction}\nConcept: {input_text}\nAssessment:",
            device, max_length=100,
        )
    if instruction_type == "concept_relation":
        related_concept = extract_keywords(input_text, n=1)[0]
        prompt = f"{instruction}\nConcept 1: {input_text}\nConcept 2: {related_concept}\nRelation:"
        return generate_gpt2_output(models["gpt2_tokenizer"], models["gpt2_model"], prompt, device, max_length=100)
    if instruction_type == "quiz_generation":
        prompt = f"{instruction}\nConcept: {input_text}\nQuestion:"
        return generate_gpt2_output(models["gpt2_tokenizer"], models["gpt2_model"], prompt, device, max_length=200)

    # concept_explanation, generate_question, provide_example, misconception, analogy, application
    prompt = f"{instruction}\nConcept: {input_text}\nAnswer:"
    return generate_gpt2_output(models["gpt2_tokenizer"], models["gpt2_model"], prompt, device, max_length=100)
