import hashlib
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class FakeSentenceModel:
    """Hashed bag-of-words embedding: identical text -> identical vector, disjoint words -> ~orthogonal."""

    dim = 64

    def _vec(self, text: str) -> torch.Tensor:
        v = torch.zeros(self.dim)
        for word in text.lower().split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            v[h] += 1.0
        if v.norm() == 0:
            v[0] = 1.0
        return v

    def encode(self, texts, convert_to_tensor=True, show_progress_bar=False):
        if isinstance(texts, str):
            return self._vec(texts)
        return torch.stack([self._vec(t) for t in texts])


class ConstantSentenceModel(FakeSentenceModel):
    """Every text maps to the same vector, so cosine similarity is always 1."""

    def _vec(self, text: str) -> torch.Tensor:
        return torch.ones(self.dim)


@pytest.fixture
def fake_model():
    return FakeSentenceModel()


@pytest.fixture
def constant_model():
    return ConstantSentenceModel()


@pytest.fixture
def config():
    from config import CONFIG
    snapshot = dict(CONFIG)
    yield CONFIG
    CONFIG.clear()
    CONFIG.update(snapshot)
