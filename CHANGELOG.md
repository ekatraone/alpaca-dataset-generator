# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Default `local` backend now uses an instruct model (`Qwen/Qwen2.5-1.5B-Instruct`) with batched chat generation; choose another with `--model` (e.g. `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`).
- `--backend legacy` keeps the original GPT-2 / T5 / DistilBERT pipeline.
- Command-line flags to override configuration (paths, counts, device) without editing source files.

### Fixed
- `--resume` with `--format sharegpt` no longer crashes during deduplication.
- Pretty-printed (multi-line) `.json` files are now loaded instead of silently yielding nothing.
- Generation makes extra passes over the input until `--num-examples` is reached.
- NLTK stopword download failure no longer breaks import.
- Validation similarity threshold now reads `min_similarity_threshold` from config.

### Changed
- Updated default paths and batch sizes in configuration for sensible out-of-the-box runs.
- README now documents CLI overrides and current defaults.

## [1.0.0] - 2024-07-02

### Added
- Initial release of the Alpaca-style Dataset Generator
- Support for processing text, PDF, and Word documents as input
- Generation of diverse instruction types: summarize, keyword extraction, title generation, sentiment analysis, question generation, and paraphrasing
- Utilization of GPT-2, T5, and BERT models for different generation tasks
- Multithreaded input file processing for improved performance
- Command-line interface for easy usage
- Comprehensive README with setup and usage instructions

### Changed

### Deprecated

### Removed

### Fixed

### Security
