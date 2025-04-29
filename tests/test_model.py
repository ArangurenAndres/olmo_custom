import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pytest

from utils.model import build_model


@pytest.fixture
def dummy_config():
    # Provide minimal dummy config to build a small test model
    return {
        "vocab_size": 1000,
        "sequence_length": 128,
        "lr": 0.001,
        "weight_decay": 0.01,
        "betas": (0.9, 0.95),
        "device": "cpu"
    }


def test_model_instantiation(dummy_config):
    model, train_module = build_model(
        vocab_size=dummy_config["vocab_size"],
        device=torch.device(dummy_config["device"]),
        sequence_length=dummy_config["sequence_length"],
        lr=dummy_config["lr"],
        weight_decay=dummy_config["weight_decay"],
        betas=dummy_config["betas"]
    )

    assert model is not None, "Model creation failed"
    assert hasattr(model, "forward"), "Model does not have a forward() method"
    assert train_module is not None, "TrainModule creation failed"


def test_model_forward(dummy_config):
    model, _ = build_model(
        vocab_size=dummy_config["vocab_size"],
        device=torch.device(dummy_config["device"]),
        sequence_length=dummy_config["sequence_length"],
        lr=dummy_config["lr"],
        weight_decay=dummy_config["weight_decay"],
        betas=dummy_config["betas"]
    )

    model.eval()

    batch_size = 2
    dummy_input = torch.randint(
        low=0,
        high=dummy_config["vocab_size"],
        size=(batch_size, dummy_config["sequence_length"]),
        dtype=torch.long,
        device=dummy_config["device"]
    )

    with torch.no_grad():
        output = model(dummy_input)

    assert isinstance(output, torch.Tensor), "Output is not a torch.Tensor"
    assert output.shape[0] == batch_size, "Batch size mismatch in output"
    assert output.shape[1] == dummy_config["sequence_length"], "Sequence length mismatch in output"
