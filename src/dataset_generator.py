import random
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from utils import generate_gpt2_output, generate_t5_output, extract_keywords, preprocess_text
from config import CONFIG

class TextDataset(Dataset):
    def __init__(self, texts, instructions):
        # Preprocess texts when loading to ensure they're within token limits
        self.texts = [preprocess_text(text) for text in texts]
        self.instructions = instructions

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        instruction_type, instruction = random.choice(self.instructions)
        return text, instruction_type, instruction

def generate_dataset(input_texts: List[str], models: Dict) -> List[Dict[str, Any]]:
    instructions = [
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

    dataset = TextDataset(input_texts, instructions)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['max_workers'])

    examples = []
    
    with tqdm(total=CONFIG['num_examples'], desc="Generating examples", unit="example") as pbar:
        for batch in dataloader:
            texts, instruction_types, instructions = batch
            batch_examples = generate_batch(models, texts, instruction_types, instructions)
            examples.extend(batch_examples)
            pbar.update(len(batch_examples))
            if len(examples) >= CONFIG['num_examples']:
                break

    return examples[:CONFIG['num_examples']]

def generate_batch(models: Dict, texts: List[str], instruction_types: List[str], instructions: List[str]) -> List[Dict[str, Any]]:
    batch_examples = []
    print(f"Generating batch of {len(texts)} examples")
    for text, instruction_type, instruction in zip(texts, instruction_types, instructions):
        if instruction_type == "summarize":
            output = generate_t5_output(models["t5_tokenizer"], models["t5_model"], "summarize", text, CONFIG['device'])
        elif instruction_type == "paraphrase":
            output = generate_t5_output(models["t5_tokenizer"], models["t5_model"], "paraphrase", text, CONFIG['device'])
        elif instruction_type == "keyword":
            keywords = extract_keywords(text)
            output = ", ".join(keywords)
        elif instruction_type == "sentiment":
            truncated_text = text[:1000]  # Approximate truncation to stay under 512 tokens
            sentiment = models["sentiment_pipeline"](truncated_text)[0]
            explanation = generate_gpt2_output(models["gpt2_tokenizer"], models["gpt2_model"], f"The text \"{truncated_text[:200]}\" has {sentiment['label']} sentiment because", CONFIG['device'])
            output = f"{sentiment['label'].capitalize()}. {explanation}"
        else:
            # instruction already contains the few-shot prefix; append the actual text as the new case
            prompt = f"{instruction}\nText: {text}\nOutput:"
            output = generate_gpt2_output(models["gpt2_tokenizer"], models["gpt2_model"], prompt, CONFIG['device'])

        batch_examples.append({
            "instruction": instruction,
            "input": text,
            "output": output,
            "instruction_type": instruction_type
        })

    return batch_examples
