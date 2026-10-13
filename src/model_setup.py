import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer, pipeline
from sentence_transformers import SentenceTransformer
from config import CONFIG


def setup_models(backend: str = 'local') -> dict:
    """
    Load models needed for the selected backend.

    For 'api' mode only the sentence-transformer (used for validation and
    deduplication) is loaded — the heavy GPT-2 / T5 / DistilBERT models are
    skipped, saving ~2 GB of memory and significant startup time.

    Args:
        backend: 'local' or 'api'

    Returns:
        dict of model objects.
    """
    models = {}

    # Sentence transformer is always needed (validation + deduplication)
    print("Loading sentence-transformer model...")
    models["sentence_model"] = SentenceTransformer(
        CONFIG['models']['sentence'], device=CONFIG['device']
    )

    if backend == 'local':
        print("Loading GPT-2...")
        models["gpt2_tokenizer"] = GPT2Tokenizer.from_pretrained(CONFIG['models']['gpt2'])
        models["gpt2_model"] = (
            GPT2LMHeadModel.from_pretrained(CONFIG['models']['gpt2']).to(CONFIG['device'])
        )
        models["gpt2_model"].config.pad_token_id = models["gpt2_model"].config.eos_token_id

        print("Loading T5...")
        models["t5_tokenizer"] = T5Tokenizer.from_pretrained(CONFIG['models']['t5'])
        models["t5_model"] = (
            T5ForConditionalGeneration.from_pretrained(CONFIG['models']['t5']).to(CONFIG['device'])
        )

        print("Loading sentiment model...")
        models["sentiment_pipeline"] = pipeline(
            "sentiment-analysis",
            model=CONFIG['models']['sentiment'],
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=512,
        )
    else:
        print("API backend selected — skipping local generation models.")

    return models
