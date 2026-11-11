import torch
from sentence_transformers import SentenceTransformer
from config import CONFIG


def setup_models(backend: str = 'local') -> dict:
    """
    Load the models needed for the selected backend.

    - 'local'  : one instruct LLM (CONFIG['local_model'], default Qwen2.5-1.5B-Instruct)
    - 'api'    : no generation models (OpenAI-compatible endpoint is used)
    - 'legacy' : the original GPT-2 / T5 / DistilBERT trio

    The sentence-transformer (validation + deduplication) is always loaded.
    """
    models = {}
    device = CONFIG['device']

    print("Loading sentence-transformer model...")
    models["sentence_model"] = SentenceTransformer(CONFIG['models']['sentence'], device=device)

    if backend == 'local':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = CONFIG['local_model']
        print(f"Loading local model {name}...")
        tokenizer = AutoTokenizer.from_pretrained(name, padding_side='left')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device).eval()
        models["llm_tokenizer"] = tokenizer
        models["llm_model"] = model

    elif backend == 'legacy':
        from transformers import (GPT2LMHeadModel, GPT2Tokenizer,
                                  T5ForConditionalGeneration, T5Tokenizer, pipeline)
        print("Loading GPT-2...")
        models["gpt2_tokenizer"] = GPT2Tokenizer.from_pretrained(CONFIG['models']['gpt2'])
        models["gpt2_model"] = GPT2LMHeadModel.from_pretrained(CONFIG['models']['gpt2']).to(device)
        models["gpt2_model"].config.pad_token_id = models["gpt2_model"].config.eos_token_id

        print("Loading T5...")
        models["t5_tokenizer"] = T5Tokenizer.from_pretrained(CONFIG['models']['t5'])
        models["t5_model"] = T5ForConditionalGeneration.from_pretrained(CONFIG['models']['t5']).to(device)

        print("Loading sentiment model...")
        models["sentiment_pipeline"] = pipeline(
            "sentiment-analysis",
            model=CONFIG['models']['sentiment'],
            device=device.index if device.type == 'cuda' and device.index is not None else (0 if device.type == 'cuda' else -1),
            truncation=True,
            max_length=512,
        )
    else:
        print("API backend selected — skipping local generation models.")

    return models
