"""Tests the LoRA API."""

from typing import cast, get_args

import pytest
import torch
from torch import Tensor, nn

import mlfab
from mlfab.nn.lora import SupportedModuleNonParallel


@pytest.mark.parametrize("mod_type", get_args(SupportedModuleNonParallel))
def test_lora_modules(mod_type: type[nn.Module]) -> None:
    """Tests loading weights from a non-LoRA model into a LoRA model.

    Args:
        mod_type: The type of the model to test.

    Raises:
        NotImplementedError: If the model type is not supported.
    """
    model: nn.Module
    lora_model: nn.Module
    in_tensor: Tensor

    if mod_type in (nn.Embedding, nn.Linear, nn.LSTM, nn.GRU):
        model = mod_type(10, 20)
        if mod_type == nn.Embedding:
            in_tensor = torch.randint(0, 10, (5, 10))
        else:
            in_tensor = torch.randn(5, 10)

    elif mod_type in (nn.LSTMCell, nn.GRUCell):
        model = mod_type(10, 20)
        in_tensor = torch.randn(5, 10)

    elif mod_type in (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d):
        model = mod_type(3, 5, 3)
        if mod_type in (nn.Conv1d, nn.ConvTranspose1d):
            in_tensor = torch.randn(5, 3, 10)
        else:
            in_tensor = torch.randn(5, 3, 10, 10)

    else:
        raise NotImplementedError(f"Unsupported model type: {mod_type}")

    lora_model = mlfab.lora(cast(SupportedModuleNonParallel, model), r=2)

    # Loads the weights from the reference model into the LoRA model.
    lora_model.load_state_dict(model.state_dict())

    # To and from eval mode to check gradient behavior.
    mlfab.reset_lora_weights_(lora_model)
    lora_model.eval()
    lora_model.train()

    ref_out, lora_out = model(in_tensor), lora_model(in_tensor)
    if isinstance(ref_out, tuple):
        ref_out, lora_out = ref_out[0], lora_out[0]

    # Checks that the outputs are initially the same for the same input.
    assert torch.allclose(ref_out, lora_out)

    lora_out.backward(torch.ones_like(lora_out))

    # Checks that the gradients for the LoRA parameters are non-null.
    for n, p in lora_model.named_parameters():
        if n.startswith("lora_") or n.startswith("bias"):
            assert p.grad is not None
        else:
            assert p.grad is None

    # Checks that the gradients for one of the LoRA parameters is non-null.
    grads = [k for k, v in lora_model.named_parameters() if v.grad is not None and v.grad.abs().sum() > 0.0]
    assert any(g.startswith("lora_") for g in grads)
