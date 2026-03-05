# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Command-line flags to override configuration (paths, counts, device) without editing source files.

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
