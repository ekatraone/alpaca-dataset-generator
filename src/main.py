import argparse
from data_loader import load_input_data
from model_setup import setup_models
from dataset_generator import generate_dataset
from validation import validate_dataset
from utils import save_to_jsonl
from config import CONFIG

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Alpaca-format instruction datasets from documents.")
    parser.add_argument("--input", type=str, help="Path to input folder containing .txt, .pdf, or .docx files")
    parser.add_argument("--output", type=str, help="Path for the raw output JSONL file")
    parser.add_argument("--validated-output", type=str, dest="validated_output", help="Path for the validated output JSONL file")
    parser.add_argument("--num-examples", type=int, dest="num_examples", help="Number of examples to generate")
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

    print("Loading input data...")
    input_texts = load_input_data(CONFIG['input_folder'])
    if not input_texts:
        print("No valid input files found. Please check your input folder.")
        return

    print("Setting up models...")
    models = setup_models()

    print(f"Generating {CONFIG['num_examples']} examples...")
    dataset = generate_dataset(input_texts, models)

    print(f"Saving raw dataset to {CONFIG['output_file']}...")
    save_to_jsonl(dataset, CONFIG['output_file'])

    print("Validating generated examples...")
    validated_dataset = validate_dataset(dataset, models["sentence_model"])

    print(f"Saving validated dataset to {CONFIG['validated_output_file']}...")
    save_to_jsonl(validated_dataset, CONFIG['validated_output_file'])

    print(f"Raw dataset with {len(dataset)} examples saved to '{CONFIG['output_file']}'")
    print(f"Validated dataset with {len(validated_dataset)} examples saved to '{CONFIG['validated_output_file']}'")

if __name__ == "__main__":
    main()
