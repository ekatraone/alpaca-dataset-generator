import argparse
from data_loader import load_input_data
from model_setup import setup_models
from validation import validate_dataset
from utils import save_to_jsonl, load_existing_jsonl, append_to_jsonl, deduplicate
from llm_backend import LLMBackend
from config import CONFIG


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Alpaca/ShareGPT instruction datasets from documents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    parser.add_argument("--input", type=str, help="Path to input folder containing source documents")
    parser.add_argument("--output", type=str, help="Path for the raw output JSONL file")
    parser.add_argument("--validated-output", type=str, dest="validated_output",
                        help="Path for the validated output JSONL file")
    parser.add_argument("--num-examples", type=int, dest="num_examples",
                        help="Number of examples to generate")
    parser.add_argument("--format", type=str, choices=["alpaca", "sharegpt"], default="alpaca",
                        dest="fmt", help="Output format")
    parser.add_argument("--mode", type=str, choices=["standard", "edu"], default="standard",
                        help="'standard': summarize/keyword/title/sentiment/question/paraphrase. "
                             "'edu': 10 educational task types (explanations, quizzes, analogies...)")

    # Backend
    parser.add_argument("--backend", type=str, choices=["local", "api"], default="local",
                        help="Generation backend: 'local' (GPT-2/T5) or 'api' (OpenRouter / any OpenAI-compatible endpoint)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model slug for API backend (e.g. 'meta-llama/llama-3.1-8b-instruct:free'). "
                             "Ignored in local mode.")
    parser.add_argument("--api-key", type=str, dest="api_key", default=None,
                        help="API key for the endpoint. Falls back to OPENROUTER_API_KEY / OPENAI_API_KEY env vars.")
    parser.add_argument("--base-url", type=str, dest="base_url", default=None,
                        help="API base URL (default: https://openrouter.ai/api/v1). "
                             "Override to use OpenAI, a local vLLM/llama-server, etc.")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Parallel requests in API mode (default 8)")

    # Chunking
    parser.add_argument("--chunk-size", type=int, dest="chunk_size",
                        help="Max characters per text chunk (default 1500)")
    parser.add_argument("--chunk-overlap", type=int, dest="chunk_overlap",
                        help="Overlap characters between chunks (default 200)")

    # Run control
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previous run — append to existing output file")
    parser.add_argument("--no-dedup", action="store_true", dest="no_dedup",
                        help="Disable semantic deduplication")

    return parser.parse_args()


def apply_overrides(args):
    overrides = {
        'input_folder': args.input,
        'output_file': args.output,
        'validated_output_file': args.validated_output,
        'num_examples': args.num_examples,
        'chunk_size': args.chunk_size,
        'chunk_overlap': args.chunk_overlap,
        'api_model': args.model,
        'api_base_url': args.base_url,
        'api_concurrency': args.concurrency,
    }
    for key, value in overrides.items():
        if value is not None:
            CONFIG[key] = value


def main():
    args = parse_args()
    apply_overrides(args)
    output_fmt = args.fmt

    if args.mode == "edu":
        from edu_dataset_generator import generate_dataset
    else:
        from dataset_generator import generate_dataset

    # ------------------------------------------------------------------
    # Resume: count already-generated examples
    # ------------------------------------------------------------------
    existing = []
    if args.resume:
        existing = load_existing_jsonl(CONFIG['output_file'])
        if existing:
            print(f"Resuming: found {len(existing)} existing examples in {CONFIG['output_file']}")

    target = CONFIG['num_examples']
    remaining = target - len(existing)
    if remaining <= 0:
        print(f"Already have {len(existing)} examples — target of {target} reached. Nothing to do.")
        print("Run without --resume or increase --num-examples to generate more.")
        return

    # ------------------------------------------------------------------
    # Load data first (cheap) so a bad input path fails before model load
    # ------------------------------------------------------------------
    print("Loading input data...")
    input_texts = load_input_data(CONFIG['input_folder'])
    if not input_texts:
        print("No valid input files found. Please check your input folder.")
        return

    # ------------------------------------------------------------------
    # Backend setup
    # ------------------------------------------------------------------
    print(f"Backend: {args.backend}  |  Mode: {args.mode}")
    models = setup_models(backend=args.backend)

    if args.backend == 'api':
        backend = LLMBackend(mode='api', models=models, api_key=args.api_key,
                             model=args.model, base_url=args.base_url)
        print(f"API model: {backend.model_name}")
        print(f"Endpoint:  {backend.base_url}")
    else:
        backend = LLMBackend(mode='local', models=models)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    print(f"Generating {remaining} new examples (target total: {target})...")
    new_examples = generate_dataset(remaining, input_texts, models, backend=backend)

    # ------------------------------------------------------------------
    # Deduplication (new examples against each other and against existing)
    # ------------------------------------------------------------------
    if not args.no_dedup:
        print("Running semantic deduplication...")
        combined = deduplicate(existing + new_examples, models["sentence_model"],
                               similarity_threshold=CONFIG.get('dedup_threshold', 0.95))
        existing_ids = {id(e) for e in existing}
        new_examples = [e for e in combined if id(e) not in existing_ids]

    # ------------------------------------------------------------------
    # Save raw output
    # ------------------------------------------------------------------
    if args.resume and existing:
        print(f"Appending {len(new_examples)} new examples to {CONFIG['output_file']}...")
        for ex in new_examples:
            append_to_jsonl(ex, CONFIG['output_file'], fmt=output_fmt)
    else:
        print(f"Saving {len(new_examples)} examples to {CONFIG['output_file']}...")
        save_to_jsonl(new_examples, CONFIG['output_file'], fmt=output_fmt)

    # ------------------------------------------------------------------
    # Validate + save validated output
    # ------------------------------------------------------------------
    print("Validating generated examples...")
    validated = validate_dataset(new_examples, models["sentence_model"])

    if args.resume and existing:
        print(f"Appending {len(validated)} validated examples to {CONFIG['validated_output_file']}...")
        for ex in validated:
            append_to_jsonl(ex, CONFIG['validated_output_file'], fmt=output_fmt)
    else:
        print(f"Saving validated dataset to {CONFIG['validated_output_file']}...")
        save_to_jsonl(validated, CONFIG['validated_output_file'], fmt=output_fmt)

    print("\nDone.")
    print(f"  Backend / mode:     {args.backend} / {args.mode}")
    print(f"  Raw examples:       {len(existing) + len(new_examples)}  ->  {CONFIG['output_file']}")
    print(f"  Validated examples: {len(validated)} new  ->  {CONFIG['validated_output_file']}")
    print(f"  Output format:      {output_fmt}")


if __name__ == "__main__":
    main()
