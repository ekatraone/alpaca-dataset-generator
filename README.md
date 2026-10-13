# Alpaca-style Dataset Generator

Turn a folder of documents into an instruction-tuning dataset. Reads `.txt`, `.pdf`, `.docx`, `.md`, `.csv`, `.json` and `.jsonl`, chunks the text, generates instruction/input/output records with either small local models or any OpenAI-compatible LLM API, then validates, deduplicates and writes JSONL in **Alpaca** or **ShareGPT** format.

## Features

- **Two generation backends**
  - `local` — GPT-2 / T5 / DistilBERT, fully offline, no API key (lower quality)
  - `api` — any OpenAI-compatible endpoint; defaults to [OpenRouter](https://openrouter.ai) (much higher quality, parallel requests)
- **Two task modes**
  - `standard` — summarize, keywords, title, sentiment, question, paraphrase
  - `edu` — concept explanation, example, analogy, misconception, quiz, learning path, concept relation, application, difficulty assessment, question
- Few-shot prompts for every task type
- Sentence-aware sliding-window chunking with overlap (no content truncated)
- Per-task validation + semantic similarity check
- Semantic near-duplicate removal
- Resume an interrupted run
- Alpaca or ShareGPT output

## Setup

```bash
git clone https://github.com/ekatraone/alpaca-dataset-generator.git
cd alpaca-dataset-generator
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

## Usage

```bash
# Local models, standard tasks
python src/main.py --input data/input --output output/raw.jsonl \
  --validated-output output/validated.jsonl --num-examples 1000

# OpenRouter (free Llama 3.1 8B by default)
export OPENROUTER_API_KEY=sk-or-...
python src/main.py --backend api --input data/input --num-examples 1000

# Pick a model, use edu tasks, emit ShareGPT
python src/main.py --backend api --model "anthropic/claude-3-haiku" \
  --mode edu --format sharegpt --input data/input --num-examples 5000

# Any OpenAI-compatible server (vLLM, llama-server, OpenAI itself)
python src/main.py --backend api --base-url http://localhost:8000/v1 \
  --model llama-3.1-8b-instruct --api-key none --input data/input

# Resume a crashed run — appends only the missing examples
python src/main.py --backend api --input data/input --num-examples 5000 --resume
```

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--input` | `config.py` | Folder of source documents (searched recursively) |
| `--output` | `config.py` | Raw output JSONL |
| `--validated-output` | `config.py` | Validated output JSONL |
| `--num-examples` | `10000` | Target number of records |
| `--mode` | `standard` | `standard` or `edu` task set |
| `--format` | `alpaca` | `alpaca` or `sharegpt` |
| `--backend` | `local` | `local` or `api` |
| `--model` | `meta-llama/llama-3.1-8b-instruct:free` | API model slug |
| `--api-key` | env | Falls back to `OPENROUTER_API_KEY` / `OPENAI_API_KEY` |
| `--base-url` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible endpoint |
| `--concurrency` | `8` | Parallel API requests |
| `--chunk-size` | `1500` | Max characters per chunk |
| `--chunk-overlap` | `200` | Overlap between chunks |
| `--resume` | off | Append to an existing output file |
| `--no-dedup` | off | Skip semantic deduplication |

Everything above can also be set in `src/config.py`, which also holds model names, sampling parameters and validation thresholds.

## Output

Alpaca (default):

```json
{"instruction": "...", "input": "<source chunk>", "output": "...", "instruction_type": "summarize"}
```

ShareGPT (`--format sharegpt`):

```json
{"conversations": [{"from": "human", "value": "<instruction>\n\n<input>"}, {"from": "gpt", "value": "<output>"}]}
```

Two files are written: the raw set and the subset that passed validation.

## How it works

1. **Load** — every supported file under `--input` is read and split into overlapping, sentence-aligned chunks.
2. **Generate** — each chunk gets an instruction type (cycled evenly) and a few-shot prompt. In `api` mode one chat completion handles every task; in `local` mode tasks are routed to T5 (summarize/paraphrase), DistilBERT (sentiment) or GPT-2.
3. **Clean** — echoed prompt labels (`Summary:`, `Answer:` …) and stray continuations are stripped.
4. **Deduplicate** — outputs are embedded with MiniLM; anything ≥ 0.95 cosine similarity to a kept record is dropped.
5. **Validate** — generic length/similarity checks plus per-type shape rules (questions end in `?`, quizzes have A–D options, etc.).

## Project structure

```
src/
  main.py                   CLI entry point
  config.py                 all defaults
  data_loader.py            file discovery + chunking
  llm_backend.py            local / API generation abstraction
  dataset_generator.py      standard task set
  edu_dataset_generator.py  educational task set
  model_setup.py            loads local models (skipped in api mode)
  validation.py             runs is_valid_output over a dataset
  utils.py                  readers, chunker, cleaning, validation, dedup, I/O
tests/                      pytest suite (no model downloads needed)
```

## Tests

```bash
pip install pytest
pytest
```

## Troubleshooting

- CUDA out-of-memory in local mode: lower `batch_size` in `config.py`, or use `--backend api`.
- `No module named 'openai'`: `pip install openai` (only needed for `--backend api`).
- Rate limits from OpenRouter: lower `--concurrency`; retries with backoff are built in.

## License

MIT — see [LICENSE](LICENSE).
