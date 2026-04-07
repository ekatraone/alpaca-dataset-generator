import argparse
import os
import sys

import torch

from config import CONFIG
from data_loader import load_input_data
from dataset_generator import generate_dataset
from model_setup import setup_models
from utils import append_to_jsonl, deduplicate, load_existing_jsonl, save_to_jsonl
from validation import validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Alpaca-format instruction datasets from documents."
    )
    parser.add_argument(
        "--input-folder",
        default=CONFIG["input_folder"],
        help="Folder containing input files (txt, pdf, docx).",
    )
    parser.add_argument(
        "--output-file",
        default=CONFIG["output_file"],
        help="Path to write the raw dataset JSONL.",
    )
    parser.add_argument(
        "--validated-output-file",
        default=CONFIG["validated_output_file"],
        help="Path to write the validated dataset JSONL.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=CONFIG["num_examples"],
        help="Number of examples to generate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=CONFIG["batch_size"],
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=CONFIG["max_workers"],
        help="Number of worker threads for data loading.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Override device selection (default auto-detects CUDA if available, else CPU).",
    )
    parser.add_argument(
        "--format",
        choices=["alpaca", "sharegpt"],
        default="alpaca",
        dest="fmt",
        help="Output format: 'alpaca' (default) or 'sharegpt'.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous run — skip examples already written to the output file.",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        dest="no_dedup",
        help="Disable semantic deduplication of generated examples.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CONFIG["chunk_size"],
        help="Max characters per text chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CONFIG["chunk_overlap"],
        help="Overlap characters between consecutive chunks.",
    )
    return parser.parse_args()


def apply_overrides(args: argparse.Namespace) -> None:
    CONFIG["input_folder"] = args.input_folder
    CONFIG["output_file"] = args.output_file
    CONFIG["validated_output_file"] = args.validated_output_file
    CONFIG["num_examples"] = args.num_examples
    CONFIG["batch_size"] = args.batch_size
    CONFIG["max_workers"] = args.max_workers
    CONFIG["chunk_size"] = args.chunk_size
    CONFIG["chunk_overlap"] = args.chunk_overlap
    if args.device:
        if args.device == "cuda" and not torch.cuda.is_available():
            print(
                "Requested CUDA but no CUDA device was detected. Falling back to CPU.",
                file=sys.stderr,
            )
            CONFIG["device"] = torch.device("cpu")
        else:
            CONFIG["device"] = torch.device(args.device)


def ensure_output_dirs() -> None:
    for key in ("output_file", "validated_output_file"):
        path = CONFIG[key]
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        # Empty directory string means current working directory - no creation needed.


def main():
    args = parse_args()
    apply_overrides(args)
    ensure_output_dirs()

    output_fmt = args.fmt

    # ------------------------------------------------------------------
    # Resume: load already-generated examples and skip that many
    # ------------------------------------------------------------------
    existing = []
    if args.resume:
        existing = load_existing_jsonl(CONFIG["output_file"])
        if existing:
            print(f"Resuming: found {len(existing)} existing examples in {CONFIG['output_file']}")

    remaining = CONFIG["num_examples"] - len(existing)
    if remaining <= 0:
        print(f"Already have {len(existing)} examples — target reached. Nothing to do.")
        print("Run without --resume or increase --num-examples to generate more.")
        return

    # ------------------------------------------------------------------
    # Load + generate
    # ------------------------------------------------------------------
    print("Loading input data...")
    input_texts = load_input_data(CONFIG["input_folder"])
    if not input_texts:
        print("No valid input files found. Please check your input folder.")
        return

    print("Setting up models...")
    models = setup_models()

    print(f"Generating {remaining} new examples (target total: {CONFIG['num_examples']})...")
    CONFIG["num_examples"] = remaining  # generate only what's still needed
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
            similarity_threshold=CONFIG.get("dedup_threshold", 0.95),
        )
        # Separate back into existing (already on disk) and new
        existing_ids = set(id(e) for e in existing)
        new_examples = [e for e in all_examples if id(e) not in existing_ids]
    else:
        all_examples = existing + new_examples

    # ------------------------------------------------------------------
    # Save raw output
    # ------------------------------------------------------------------
    if args.resume and existing:
        print(f"Appending {len(new_examples)} new examples to {CONFIG['output_file']}...")
        for ex in new_examples:
            append_to_jsonl(ex, CONFIG["output_file"], fmt=output_fmt)
    else:
        print(f"Saving {len(all_examples)} examples to {CONFIG['output_file']}...")
        save_to_jsonl(all_examples, CONFIG["output_file"], fmt=output_fmt)

    # ------------------------------------------------------------------
    # Validate and save validated output
    # ------------------------------------------------------------------
    print("Validating generated examples...")
    validated = validate_dataset(new_examples, models["sentence_model"])

    if args.resume and existing:
        print(f"Appending {len(validated)} validated examples to {CONFIG['validated_output_file']}...")
        for ex in validated:
            append_to_jsonl(ex, CONFIG["validated_output_file"], fmt=output_fmt)
    else:
        print(f"Saving validated dataset to {CONFIG['validated_output_file']}...")
        save_to_jsonl(validated, CONFIG["validated_output_file"], fmt=output_fmt)

    total_raw = len(existing) + len(new_examples)
    print("\nDone.")
    print(f"  Raw examples:       {total_raw} (in '{CONFIG['output_file']}')")
    print(f"  Validated examples: {len(validated)} new (in '{CONFIG['validated_output_file']}')")
    print(f"  Output format:      {output_fmt}")


if __name__ == "__main__":
    main()
