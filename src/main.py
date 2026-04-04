import argparse
from data_loader import load_input_data
from model_setup import setup_models
from dataset_generator import generate_dataset
from validation import validate_dataset
from utils import save_to_jsonl, load_existing_jsonl, append_to_jsonl, deduplicate
from config import CONFIG

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Alpaca-format instruction datasets from documents.")
    parser.add_argument("--input", type=str, help="Path to input folder containing source documents")
    parser.add_argument("--output", type=str, help="Path for the raw output JSONL file")
    parser.add_argument("--validated-output", type=str, dest="validated_output", help="Path for the validated output JSONL file")
    parser.add_argument("--num-examples", type=int, dest="num_examples", help="Number of examples to generate")
    parser.add_argument("--format", type=str, choices=["alpaca", "sharegpt"], default="alpaca", dest="fmt",
                        help="Output format: 'alpaca' (default) or 'sharegpt'")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previous run — skip examples already written to the output file")
    parser.add_argument("--no-dedup", action="store_true", dest="no_dedup",
                        help="Disable semantic deduplication of generated examples")
    parser.add_argument("--chunk-size", type=int, dest="chunk_size", help="Max characters per text chunk (default 1500)")
    parser.add_argument("--chunk-overlap", type=int, dest="chunk_overlap", help="Overlap characters between chunks (default 200)")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.input:
        CONFIG['input_folder'] = args.input
    if args.output:
        CONFIG['output_file'] = args.output
    if args.validated_output:
        CONFIG['validated_output_file'] = args.validated_output
    if args.num_examples:
        CONFIG['num_examples'] = args.num_examples
    if args.chunk_size:
        CONFIG['chunk_size'] = args.chunk_size
    if args.chunk_overlap:
        CONFIG['chunk_overlap'] = args.chunk_overlap

    output_fmt = args.fmt

    # ------------------------------------------------------------------
    # Resume: load already-generated examples and skip that many
    # ------------------------------------------------------------------
    existing = []
    if args.resume:
        existing = load_existing_jsonl(CONFIG['output_file'])
        if existing:
            print(f"Resuming: found {len(existing)} existing examples in {CONFIG['output_file']}")

    remaining = CONFIG['num_examples'] - len(existing)
    if remaining <= 0:
        print(f"Already have {len(existing)} examples — target reached. Nothing to do.")
        print("Run without --resume or increase --num-examples to generate more.")
        return

    # ------------------------------------------------------------------
    # Load + generate
    # ------------------------------------------------------------------
    print("Loading input data...")
    input_texts = load_input_data(CONFIG['input_folder'])
    if not input_texts:
        print("No valid input files found. Please check your input folder.")
        return

    print("Setting up models...")
    models = setup_models()

    print(f"Generating {remaining} new examples (target total: {CONFIG['num_examples']})...")
    CONFIG['num_examples'] = remaining  # generate only what's still needed
    new_examples = generate_dataset(input_texts, models)

    # ------------------------------------------------------------------
    # Deduplication (across new + existing)
    # ------------------------------------------------------------------
    if not args.no_dedup:
        print("Running semantic deduplication...")
        all_examples = existing + new_examples
        all_examples = deduplicate(
            all_examples,
            models["sentence_model"],
            similarity_threshold=CONFIG.get('dedup_threshold', 0.95),
        )
        # Separate back into existing (already on disk) and new
        existing_set = set(id(e) for e in existing)
        new_examples = [e for e in all_examples if id(e) not in existing_set]
    else:
        all_examples = existing + new_examples

    # ------------------------------------------------------------------
    # Save raw output
    # ------------------------------------------------------------------
    if args.resume and existing:
        # Append only the new examples; existing are already in the file
        print(f"Appending {len(new_examples)} new examples to {CONFIG['output_file']}...")
        for ex in new_examples:
            append_to_jsonl(ex, CONFIG['output_file'], fmt=output_fmt)
    else:
        print(f"Saving {len(all_examples)} examples to {CONFIG['output_file']}...")
        save_to_jsonl(all_examples, CONFIG['output_file'], fmt=output_fmt)

    # ------------------------------------------------------------------
    # Validate and save validated output
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

    total_raw = len(existing) + len(new_examples)
    print(f"\nDone.")
    print(f"  Raw examples:       {total_raw} (in '{CONFIG['output_file']}')")
    print(f"  Validated examples: {len(validated)} new (in '{CONFIG['validated_output_file']}')")
    print(f"  Output format:      {output_fmt}")

if __name__ == "__main__":
    main()
