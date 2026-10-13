import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from utils import (
    generate_gpt2_output, generate_t5_output, extract_keywords, preprocess_text, clean_output,
)
from config import CONFIG


INSTRUCTIONS: List[Tuple[str, str]] = [
    ("summarize",
     "Provide a concise one-sentence summary of the following text.\n\n"
     "Example:\n"
     "Text: The mitochondria generate energy for the cell by converting nutrients into ATP through a process called cellular respiration. This process requires oxygen and produces carbon dioxide as a byproduct.\n"
     "Summary: Mitochondria produce ATP — the cell's energy currency — by converting nutrients through oxygen-dependent cellular respiration.\n\n"
     "Now summarize:"),

    ("keyword",
     "Extract 3-5 main keywords or key phrases from the following text. Return them as a comma-separated list.\n\n"
     "Example:\n"
     "Text: Deep learning models use multiple layers of artificial neurons to learn hierarchical representations of data, enabling breakthroughs in image recognition and natural language processing.\n"
     "Keywords: deep learning, artificial neurons, hierarchical representations, image recognition, natural language processing\n\n"
     "Now extract keywords from:"),

    ("title",
     "Generate a short, engaging title (5-8 words) for the following text.\n\n"
     "Example:\n"
     "Text: Studies show that regular aerobic exercise improves memory, reduces anxiety, and may slow cognitive decline in older adults by promoting neuroplasticity and increasing blood flow to the brain.\n"
     "Title: How Exercise Rewires and Protects the Brain\n\n"
     "Now generate a title for:"),

    ("sentiment",
     "Analyze the sentiment of the following text. Classify it as Positive, Negative, or Neutral, then briefly explain your reasoning in one sentence.\n\n"
     "Example:\n"
     "Text: The product arrived two days late and the packaging was damaged, though the item itself worked fine.\n"
     "Sentiment: Negative. Despite the product functioning correctly, the late delivery and damaged packaging create an overall negative customer experience.\n\n"
     "Now analyze the sentiment of:"),

    ("question",
     "Generate one thought-provoking question based on the main idea of the following text. The question should end with a question mark.\n\n"
     "Example:\n"
     "Text: Antibiotic resistance is accelerating globally as bacteria evolve faster than new drugs are developed, threatening to make common infections untreatable.\n"
     "Question: If antibiotic resistance continues to outpace drug development, how should healthcare systems prioritize access to the remaining effective antibiotics?\n\n"
     "Now generate a question based on:"),

    ("paraphrase",
     "Rewrite the following text in different words while preserving its core meaning. Do not add new information.\n\n"
     "Example:\n"
     "Text: The stock market experienced significant volatility last quarter due to rising interest rates and investor uncertainty about inflation.\n"
     "Paraphrase: Last quarter's stock market saw sharp fluctuations driven by higher interest rates and widespread investor concern over inflation levels.\n\n"
     "Now paraphrase:"),
]


def generate_dataset(
    num_examples: int,
    input_texts: List[str],
    models: Dict,
    backend=None,
) -> List[Dict[str, Any]]:
    """Generate `num_examples` Alpaca records, cycling instruction types evenly over random chunks."""
    texts = [preprocess_text(t) for t in input_texts]
    jobs = []
    for i in range(num_examples):
        instruction_type, instruction = INSTRUCTIONS[i % len(INSTRUCTIONS)]
        jobs.append((random.choice(texts), instruction_type, instruction))
    random.shuffle(jobs)

    examples: List[Dict[str, Any]] = []
    batch_size = CONFIG['batch_size']
    with tqdm(total=num_examples, desc="Generating examples", unit="example") as pbar:
        for start in range(0, len(jobs), batch_size):
            batch = jobs[start:start + batch_size]
            examples.extend(generate_batch(models, batch, backend))
            pbar.update(len(batch))
    return examples


def generate_batch(
    models: Dict,
    jobs: List[Tuple[str, str, str]],
    backend=None,
) -> List[Dict[str, Any]]:
    """Generate outputs for a batch of (text, instruction_type, instruction) jobs."""
    def run(job):
        text, instruction_type, instruction = job
        output = clean_output(_generate_output(models, backend, instruction_type, instruction, text))
        return {
            "instruction": instruction,
            "input": text,
            "output": output,
            "instruction_type": instruction_type,
        }

    if backend is not None and backend.is_api():
        with ThreadPoolExecutor(max_workers=CONFIG.get('api_concurrency', 8)) as pool:
            return list(pool.map(run, jobs))
    return [run(job) for job in jobs]


def _generate_output(models, backend, instruction_type: str, instruction: str, text: str) -> str:
    """Route generation to the appropriate backend / model."""

    # ---- API backend: single unified call for everything ----
    if backend is not None and backend.is_api():
        if instruction_type == "sentiment":
            return backend.generate_sentiment(text[:1000])
        prompt = f"{instruction}\nText: {text}\nOutput:"
        return backend.generate(prompt, max_tokens=CONFIG.get('api_max_tokens', 200))

    # ---- Local backend: existing model routing ----
    device = CONFIG['device']
    if instruction_type == "summarize":
        return generate_t5_output(models["t5_tokenizer"], models["t5_model"], "summarize", text, device)
    if instruction_type == "paraphrase":
        return generate_t5_output(models["t5_tokenizer"], models["t5_model"], "paraphrase", text, device)
    if instruction_type == "keyword":
        return ", ".join(extract_keywords(text))
    if instruction_type == "sentiment":
        truncated = text[:1000]
        sentiment = models["sentiment_pipeline"](truncated)[0]
        explanation = generate_gpt2_output(
            models["gpt2_tokenizer"], models["gpt2_model"],
            f"The text \"{truncated[:200]}\" has {sentiment['label']} sentiment because",
            device,
        )
        return f"{sentiment['label'].capitalize()}. {explanation}"

    # title, question — GPT-2 with few-shot prompt
    prompt = f"{instruction}\nText: {text}\nOutput:"
    return generate_gpt2_output(models["gpt2_tokenizer"], models["gpt2_model"], prompt, device)
