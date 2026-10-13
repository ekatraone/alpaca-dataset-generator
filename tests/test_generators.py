from collections import Counter

import pytest

import dataset_generator
import edu_dataset_generator
from data_loader import load_input_data


CANNED = {
    "summarize": "Summary: Plants convert light energy into chemical energy inside their cells efficiently.",
    "keyword": "Keywords: photosynthesis, light energy, chemical energy, plant cells",
    "title": "Title: How Plants Turn Light Into Fuel",
    "question": "Question: Why do plants need light to make chemical energy?",
    "paraphrase": "Paraphrase: Inside plant cells, photosynthesis changes light energy into chemical energy.",
    "concept_explanation": "Answer: " + "Photosynthesis is how plants eat light. " * 8,
    "generate_question": "What limits photosynthesis in low light?",
    "provide_example": "A leaf in sunlight makes sugar from air and water all day long.",
    "learning_path": "1. Cell biology basics\n2. Light reactions\n3. Calvin cycle\n4. Advanced ecology",
    "misconception": "People think plants eat soil, but they build mass mostly from carbon dioxide in air.",
    "analogy": "A leaf is a solar panel that stores electricity as sugar instead of in a battery.",
    "quiz_generation": "What powers photosynthesis?\nA) Soil\nB) Sunlight [CORRECT]\nC) Wind\nD) Rocks",
    "concept_relation": "Photosynthesis and respiration are inverse processes that cycle carbon and energy.",
    "application": "Greenhouse growers tune light spectra to boost crop yields using this principle.",
    "difficulty_assessment": "Intermediate. Students struggle with the two-stage reaction model and its inputs.",
}

# Map the first words of each prompt to the instruction type it belongs to.
PROMPT_PREFIX = {
    "Provide a concise one-sentence summary": "summarize",
    "Extract 3-5 main keywords": "keyword",
    "Generate a short, engaging title": "title",
    "Generate one thought-provoking question based on the main idea": "question",
    "Rewrite the following text": "paraphrase",
    "Explain the following concept": "concept_explanation",
    "Generate a thought-provoking question about": "generate_question",
    "Provide a concrete, real-world example": "provide_example",
    "Suggest a step-by-step learning path": "learning_path",
    "Identify a common misconception": "misconception",
    "Create a clear analogy": "analogy",
    "Generate a multiple-choice quiz": "quiz_generation",
    "Explain how the following concept relates": "concept_relation",
    "Describe a specific, practical application": "application",
    "Assess the difficulty level": "difficulty_assessment",
}


class FakeApiBackend:
    def __init__(self):
        self.prompts = []

    def is_api(self):
        return True

    def generate(self, prompt, max_tokens=200):
        self.prompts.append(prompt)
        for prefix, itype in PROMPT_PREFIX.items():
            if prompt.startswith(prefix):
                return CANNED[itype]
        raise AssertionError(f"unrecognised prompt: {prompt[:60]}")

    def generate_sentiment(self, text):
        self.prompts.append(text)
        return "Sentiment: Positive. The passage describes a beneficial process."


TEXTS = [
    "Photosynthesis converts light energy into chemical energy inside plant cells. It happens in chloroplasts. Oxygen is released.",
    "Cellular respiration releases energy stored in glucose. It occurs in mitochondria. Carbon dioxide is produced.",
]


def test_standard_generator_balances_types_and_cleans_labels(config):
    config["batch_size"] = 4
    backend = FakeApiBackend()
    examples = dataset_generator.generate_dataset(12, TEXTS, models={}, backend=backend)

    assert len(examples) == 12
    counts = Counter(e["instruction_type"] for e in examples)
    assert set(counts) == {t for t, _ in dataset_generator.INSTRUCTIONS}
    assert all(c == 2 for c in counts.values())
    assert all(not e["output"].lower().startswith(("summary:", "keywords:", "title:", "sentiment:")) for e in examples)
    assert all(e["input"] in TEXTS for e in examples)
    assert len(backend.prompts) == 12


def test_edu_generator_validates_and_balances(config, constant_model):
    config["min_similarity_threshold"] = 0.0
    backend = FakeApiBackend()
    examples = edu_dataset_generator.generate_dataset(10, TEXTS, models={"sentence_model": constant_model}, backend=backend)

    assert len(examples) == 10
    assert Counter(e["instruction_type"] for e in examples) == {t: 1 for t, _ in config["instruction_types"]}
    assert all("instruction_type" in e and e["output"] for e in examples)
    assert not any(e["output"].startswith("Answer:") for e in examples)


def test_edu_generator_drops_slots_that_never_validate(config, constant_model):
    class BadBackend(FakeApiBackend):
        def generate(self, prompt, max_tokens=200):
            return "x"  # always fails validation

    examples = edu_dataset_generator.generate_dataset(3, TEXTS, models={"sentence_model": constant_model}, backend=BadBackend())
    assert examples == []


def test_load_input_data_reads_all_supported_types(tmp_path, config):
    config["max_workers"] = 2
    config["chunk_size"], config["chunk_overlap"] = 200, 40
    (tmp_path / "a.txt").write_text(" ".join(f"Line {i} of plain text content goes right here." for i in range(20)))
    (tmp_path / "b.md").write_text("# Heading\n\n" + " ".join(f"Markdown sentence {i} with enough words to count." for i in range(20)))
    (tmp_path / "c.csv").write_text("col1,col2\n" + "\n".join(f"value number {i} here,another value {i} here" for i in range(10)))
    (tmp_path / "d.json").write_text('[{"title": "Entry one about something long enough", "body": "Body text that is reasonably long for a chunk."}]')
    (tmp_path / "ignored.xyz").write_text("should be skipped")
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "e.jsonl").write_text('{"text": "Nested jsonl record with a good amount of text in it to pass."}\n')

    chunks = load_input_data(str(tmp_path))
    assert len(chunks) >= 6
    assert any("Markdown sentence" in c for c in chunks)
    assert any("col1: value number" in c for c in chunks)
    assert any("Nested jsonl" in c for c in chunks)
    assert not any("skipped" in c for c in chunks)
