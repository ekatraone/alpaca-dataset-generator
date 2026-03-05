# Alpaca-style Dataset Generator

This project generates a high-quality Alpaca-style dataset from input text files, PDFs, and Word documents. It features optimized performance, GPU acceleration, and customizable output.

## Features

- Multi-threaded data loading from various file formats (txt, pdf, docx)
- Batch processing for efficient dataset generation
- GPU acceleration (if available)
- Separate raw and validated output files
- Progress tracking for all major steps
- Customizable configuration

## Project Structure

```
alpaca-dataset-generator/
│
├── src/
│   ├── main.py
│   ├── config.py
│   ├── data_loader.py
│   ├── model_setup.py
│   ├── dataset_generator.py
│   ├── validation.py
│   └── utils.py
│
├── data/
│   └── input/
│       ├── file1.txt
│       ├── file2.pdf
│       └── file3.docx
│
├── output/
│   ├── raw_dataset.jsonl
│   └── validated_dataset.jsonl
│
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/ekatraone/alpaca-dataset-generator.git
cd alpaca-dataset-generator
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Download NLTK data:

```bash
python -m nltk.downloader punkt stopwords
```

## Configuration

Defaults live in `src/config.py` and can be overridden at runtime via CLI flags:

- `input_folder`: Path to your input data folder (default: `data/input`)
- `output_file`: Path for the raw output file (default: `output/raw_dataset.jsonl`)
- `validated_output_file`: Path for the validated output file (default: `output/validated_dataset.jsonl`)
- `num_examples`: Number of examples to generate (default: `1000`)
- `batch_size`: Batch size for processing (default: `8`)
- `max_workers`: Number of worker threads for data loading (default: `2`)
- `device`: Auto-detected (CUDA if available), override with `--device cpu` or `--device cuda`

## Usage

1. Place your input files (.txt, .pdf, .docx) in the `data/input/` directory (or pass `--input-folder` to point elsewhere).

2. Run the script, overriding defaults as needed:

```bash
python src/main.py \
  --num-examples 200 \
  --input-folder custom_data/input \
  --output-file output/raw_dataset.jsonl \
  --validated-output-file output/validated_dataset.jsonl \
  --device cpu
```

3. The script will generate two files in the `output/` directory (creating the folder if it does not exist):
   - `raw_dataset.jsonl`: Contains all generated examples
   - `validated_dataset.jsonl`: Contains only the examples that passed validation

## Customization

- To modify the types of examples generated, edit the `instructions` list in `src/dataset_generator.py`.
- To adjust validation criteria, modify the `is_valid_output` function in `src/utils.py`.

## Troubleshooting

- If you encounter CUDA out-of-memory errors, try reducing the `batch_size` in `src/config.py`.
- If the process is too slow, you can try increasing `max_workers` or `batch_size`, but be cautious of memory usage.

## Releases

For information about the latest releases and changes, please refer to the CHANGELOG.md file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
