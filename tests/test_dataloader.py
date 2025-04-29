import sys
import os
import pytest
import shutil
import numpy as np
from unittest.mock import patch

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataloader import prepare_data


@pytest.fixture
def test_data_dir(tmp_path):
    """Creates a temporary directory for test files."""
    return str(tmp_path / "test_data")


@pytest.fixture
def mock_load_dataset():
    """Mocks the load_dataset function to return a fake dataset."""
    class MockDataset:
        def __init__(self):
            # Create dummy articles with enough tokens
            self.data = [{"text": "This is a dummy article. " * 300}] * 10  # 300 repetitions

        def select(self, indices):
            return self  # just return itself (simple mock behavior)

        def __iter__(self):
            return iter(self.data)

        def __len__(self):
            return len(self.data)

    with patch("utils.dataloader.load_dataset", return_value=MockDataset()):
        yield


def test_prepare_data_runs(test_data_dir, mock_load_dataset):
    """Tests that prepare_data works correctly."""
    loader, tokenizer_config = prepare_data(
        data_dir=test_data_dir,
        total_sequences=5,
        sequence_length=16,
        use_small_dataset=True
    )

    # IMPORTANT: call reshuffle() before using OLMo-core loader
    loader.reshuffle(epoch=1)

    # ==== Check types ====
    assert loader is not None, "Loader is None"
    assert tokenizer_config is not None, "Tokenizer config is None"

    # ==== Check batch ====
    batch = next(iter(loader))
    assert batch is not None, "No batch returned from dataloader"
    assert "input_ids" in batch, "Batch missing 'input_ids'"
    assert batch["input_ids"].shape[1] == 16, "Incorrect sequence length in batch"

    # ==== Check token file exists ====
    token_file = os.path.join(test_data_dir, "wiki_tokens.npy")
    assert os.path.exists(token_file), "Token file not created"

    # ==== Check token file shape ====
    tokens = np.load(token_file)
    assert tokens.shape[1] == 16, "Token file shape mismatch"
