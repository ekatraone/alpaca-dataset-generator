import argparse
import os

import torch

from config import CONFIG
from data_loader import load_input_data
from dataset_generator import generate_dataset
from model_setup import setup_models
from utils import save_to_jsonl
from validation import validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Alpaca-style datasets from text, PDF, and DOCX inputs."
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
        help="Override device selection (default auto-detect).",
    )
    return parser.parse_args()


def apply_overrides(args: argparse.Namespace) -> None:
    CONFIG["input_folder"] = args.input_folder
    CONFIG["output_file"] = args.output_file
    CONFIG["validated_output_file"] = args.validated_output_file
    CONFIG["num_examples"] = args.num_examples
    CONFIG["batch_size"] = args.batch_size
    CONFIG["max_workers"] = args.max_workers
    if args.device:
        CONFIG["device"] = torch.device(
            "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
        )


def ensure_output_dirs() -> None:
    for key in ("output_file", "validated_output_file"):
        path = CONFIG[key]
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)


def main():
    args = parse_args()
    apply_overrides(args)
    ensure_output_dirs()

    print("Loading input data...")
    input_texts = load_input_data(CONFIG["input_folder"])
    if not input_texts:
        print("No valid input files found. Please check your input folder.")
        return

    print("Setting up models...")
    models = setup_models()

    print(f"Generating {CONFIG['num_examples']} examples...")
    dataset = generate_dataset(input_texts, models)

    print(f"Saving raw dataset to {CONFIG['output_file']}...")
    save_to_jsonl(dataset, CONFIG["output_file"])

    print("Validating generated examples...")
    validated_dataset = validate_dataset(dataset, models["sentence_model"])

    print(f"Saving validated dataset to {CONFIG['validated_output_file']}...")
    save_to_jsonl(validated_dataset, CONFIG["validated_output_file"])

    print(
        f"Raw dataset with {len(dataset)} examples saved to '{CONFIG['output_file']}'"
    )
    print(
        f"Validated dataset with {len(validated_dataset)} examples saved to '{CONFIG['validated_output_file']}'"
    )


if __name__ == "__main__":
    main()
