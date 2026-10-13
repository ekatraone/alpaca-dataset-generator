import json

import pytest

import utils
from utils import (
    chunk_text, clean_output, deduplicate, from_sharegpt, is_valid_output,
    load_existing_jsonl, read_file, save_to_jsonl, append_to_jsonl, to_sharegpt,
    validation_failure,
)


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

def test_chunk_text_keeps_all_content_and_respects_size():
    sentences = [f"Sentence number {i} is here and has some words in it." for i in range(60)]
    text = " ".join(sentences)
    chunks = chunk_text(text, chunk_size=300, overlap=60)

    assert len(chunks) > 1
    assert all(len(c) <= 300 + 60 for c in chunks)
    for s in sentences:
        assert any(s in c for c in chunks), f"lost: {s}"


def test_chunk_text_overlaps_consecutive_chunks():
    text = " ".join(f"Point {i} matters a great deal to everyone involved." for i in range(40))
    chunks = chunk_text(text, chunk_size=250, overlap=100)
    first_sentences = lambda c: c.split(". ")[0]
    for prev, nxt in zip(chunks, chunks[1:]):
        assert first_sentences(nxt) in prev


def test_chunk_text_drops_tiny_fragments_and_handles_empty():
    assert chunk_text("") == []
    assert chunk_text("Too short.") == []


def test_chunk_text_normalises_whitespace():
    chunks = chunk_text("Hello    world  this\n\nis   spaced out a lot. " * 5, chunk_size=1000)
    assert "    " not in chunks[0]


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------

def test_read_csv_rows_become_field_value_chunks(tmp_path):
    p = tmp_path / "rows.csv"
    p.write_text("name,role\nAda,engineer\nGrace,admiral\n", encoding="utf-8")
    text = read_file(str(p))
    assert "name: Ada; role: engineer" in text
    assert "name: Grace; role: admiral" in text
    assert text.count("\n\n") == 1


def test_read_json_list_and_jsonl(tmp_path):
    j = tmp_path / "data.json"
    j.write_text(json.dumps([{"a": 1, "b": {"c": "x"}}, {"a": 2}]), encoding="utf-8")
    text = read_file(str(j))
    assert "a: 1" in text and "b.c: x" in text and "a: 2" in text

    jl = tmp_path / "data.jsonl"
    jl.write_text('{"k": "v1"}\n{"k": "v2"}\n', encoding="utf-8")
    assert read_file(str(jl)).split("\n\n") == ["k: v1", "k: v2"]


def test_read_markdown_strips_code_and_headings(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nSome prose here.\n\n```python\nprint('hi')\n```\n\nMore `inline` text.\n", encoding="utf-8")
    text = read_file(str(p))
    assert "Title" in text and "# Title" not in text
    assert "print" not in text
    assert "`" not in text


def test_read_file_rejects_unknown_extension(tmp_path):
    p = tmp_path / "x.xyz"
    p.write_text("nope")
    with pytest.raises(ValueError):
        read_file(str(p))


# ---------------------------------------------------------------------------
# output formats + resume helpers
# ---------------------------------------------------------------------------

EXAMPLE = {
    "instruction": "Summarize this.",
    "input": "Some input text.",
    "output": "A summary.",
    "instruction_type": "summarize",
}


def test_sharegpt_round_trip():
    sg = to_sharegpt(EXAMPLE)
    assert [t["from"] for t in sg["conversations"]] == ["human", "gpt"]
    assert sg["conversations"][0]["value"] == "Summarize this.\n\nSome input text."
    back = from_sharegpt(sg)
    assert back["instruction"] == EXAMPLE["instruction"]
    assert back["input"] == EXAMPLE["input"]
    assert back["output"] == EXAMPLE["output"]


def test_load_existing_handles_both_formats_and_bad_lines(tmp_path):
    p = tmp_path / "out" / "raw.jsonl"
    save_to_jsonl([EXAMPLE], str(p), fmt="alpaca")
    append_to_jsonl(EXAMPLE, str(p), fmt="sharegpt")
    with open(p, "a") as f:
        f.write("not json\n\n")

    records = load_existing_jsonl(str(p))
    assert len(records) == 2
    assert all("output" in r for r in records)


def test_load_existing_missing_file():
    assert load_existing_jsonl("/nonexistent/file.jsonl") == []


# ---------------------------------------------------------------------------
# clean_output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Summary: Cells make energy.", "Cells make energy."),
    ("  Keywords: a, b, c  ", "a, b, c"),
    ('"Quoted title"', "Quoted title"),
    ("Learning path:\n1. Basics\n2. More", "1. Basics\n2. More"),
    ("Good answer.\nText: the model started a new example", "Good answer."),
    ("Good answer.\nNow summarize: something", "Good answer."),
    ("No label here", "No label here"),
])
def test_clean_output(raw, expected):
    assert clean_output(raw) == expected


# ---------------------------------------------------------------------------
# is_valid_output
# ---------------------------------------------------------------------------

INPUT = "Photosynthesis converts light energy into chemical energy inside plant cells."


@pytest.mark.parametrize("itype,output,ok", [
    ("summarize", "Plants turn light into chemical energy through photosynthesis in their cells.", True),
    ("summarize", "Too short here.", False),
    ("keyword", "photosynthesis, light energy, chemical energy, plant cells", True),
    ("keyword", "photosynthesis, energy", False),
    ("title", "How Plants Turn Light Into Fuel", True),
    ("title", "Plants", False),
    ("sentiment", "Positive. The passage describes a beneficial process.", True),
    ("sentiment", "The passage is upbeat and positive overall.", False),
    ("question", "Why do plants need light to make chemical energy?", True),
    ("question", "Plants need light to make energy.", False),
    ("paraphrase", "Inside plant cells, photosynthesis changes light energy into chemical energy.", True),
    ("paraphrase", INPUT, False),
    ("generate_question", "What limits photosynthesis in low light?", True),
    ("quiz_generation", "What powers photosynthesis?\nA) Soil\nB) Sunlight [CORRECT]\nC) Wind\nD) Rocks", True),
    ("quiz_generation", "What powers photosynthesis? A) Soil B) Sunlight", False),
    ("learning_path", "1. Cell biology basics\n2. Light reactions\n3. Calvin cycle\n4. Advanced topics", True),
    ("learning_path", "Just read a book about it and you will be fine.", False),
    ("difficulty_assessment", "Intermediate. Students struggle with the two-stage reaction model.", True),
    ("difficulty_assessment", "It is fine. Most people get it quickly with some practice.", False),
])
def test_is_valid_output_shape_rules(constant_model, itype, output, ok):
    assert is_valid_output(itype, output, INPUT, constant_model) is ok


def test_is_valid_output_rejects_low_similarity(fake_model, config):
    config['min_similarity_threshold'] = 0.3
    unrelated = "Quarterly revenue grew because marketing spend increased substantially."
    assert is_valid_output("summarize", unrelated + " More words to pass length.", INPUT, fake_model) is False


def test_is_valid_output_rejects_empty(constant_model):
    assert is_valid_output("summarize", "", INPUT, constant_model) is False


@pytest.mark.parametrize("itype,output,reason", [
    ("summarize", "", "empty"),
    ("summarize", "tiny", "too_short"),
    ("summarize", "Way too short.", "summary_length"),
    ("question", "No question mark here at all.", "not_a_question"),
    ("quiz_generation", "What? A) one B) two", "quiz_shape"),
    ("sentiment", "Positive. Clearly upbeat text overall.", None),
])
def test_validation_failure_reasons(constant_model, itype, output, reason):
    assert validation_failure(itype, output, INPUT, constant_model) == reason


def test_validation_failure_low_similarity(fake_model, config):
    config['min_similarity_threshold'] = 0.3
    unrelated = "Quarterly revenue grew because marketing spend increased substantially again."
    assert validation_failure("summarize", unrelated, INPUT, fake_model) == "low_similarity"


# ---------------------------------------------------------------------------
# deduplicate
# ---------------------------------------------------------------------------

def test_deduplicate_removes_near_duplicates_keeps_first(fake_model):
    ex = lambda out: {**EXAMPLE, "output": out}
    examples = [
        ex("the quick brown fox jumps over the lazy dog"),
        ex("the quick brown fox jumps over the lazy dog"),      # exact dup
        ex("completely different words about finance markets"),
        ex("the quick brown fox jumps over the lazy dog again"),  # near dup
    ]
    kept = deduplicate(examples, fake_model, similarity_threshold=0.9)
    assert [k["output"] for k in kept] == [examples[0]["output"], examples[2]["output"]]


def test_deduplicate_empty(fake_model):
    assert deduplicate([], fake_model) == []
