import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "vibevoice" not in sys.modules:
    import types

    vibevoice_pkg = types.ModuleType("vibevoice")
    modular_pkg = types.ModuleType("vibevoice.modular")
    modeling_pkg = types.ModuleType("vibevoice.modular.modeling_vibevoice")
    configuration_pkg = types.ModuleType("vibevoice.modular.configuration_vibevoice")
    processor_pkg = types.ModuleType("vibevoice.processor")
    processor_subpkg = types.ModuleType("vibevoice.processor.vibevoice_processor")
    data_pkg = types.ModuleType("data_vibevoice")

    class _StubModel(nn.Module):
        pass

    class _StubConfig:
        pass

    class _StubProcessor:
        tokenizer = None

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):  # pragma: no cover - test shim
            return cls()

    class _StubDataset:
        pass

    class _StubCollator:
        pass

    modeling_pkg.VibeVoiceForConditionalGeneration = _StubModel
    configuration_pkg.VibeVoiceConfig = _StubConfig
    processor_subpkg.VibeVoiceProcessor = _StubProcessor
    data_pkg.VibeVoiceDataset = _StubDataset
    data_pkg.VibeVoiceCollator = _StubCollator

    sys.modules["vibevoice"] = vibevoice_pkg
    sys.modules["vibevoice.modular"] = modular_pkg
    sys.modules["vibevoice.modular.modeling_vibevoice"] = modeling_pkg
    sys.modules["vibevoice.modular.configuration_vibevoice"] = configuration_pkg
    sys.modules["vibevoice.processor"] = processor_pkg
    sys.modules["vibevoice.processor.vibevoice_processor"] = processor_subpkg
    sys.modules["data_vibevoice"] = data_pkg

from src.finetune_vibevoice_lora import (
    _cast_module_to_dtype,
    _ensure_lora_params_trainable,
    _safe_save_state_dict,
)


@pytest.mark.skipif("CI" in os.environ, reason="Quantized smoke test is lightweight but optional for CI runs.")
def test_lora_requires_grad_and_fp_modules(tmp_path):
    bitsandbytes = pytest.importorskip("bitsandbytes")

    class DummyLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = bitsandbytes.nn.Linear4bit(8, 8, bias=False)

        def forward(self, x):
            return self.linear(x)

    class DummyCore(nn.Module):
        def __init__(self):
            super().__init__()
            lora_cfg = LoraConfig(r=4, lora_alpha=16, lora_dropout=0.0, bias="none", target_modules=["linear"])
            self.language_model = get_peft_model(DummyLanguageModel(), lora_cfg)
            self.prediction_head = nn.Linear(8, 8)
            self.acoustic_connector = nn.Linear(4, 4)
            self.semantic_connector = nn.Linear(4, 4)

    class DummyWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = DummyCore()

    dummy = DummyWrapper()

    for _, param in dummy.named_parameters():
        param.requires_grad = False

    _ensure_lora_params_trainable(dummy)

    lora_params = {name: p for name, p in dummy.named_parameters() if "lora_" in name}
    assert lora_params, "LoRA parameters should be present in the dummy language model"
    assert all(param.requires_grad for param in lora_params.values())

    _cast_module_to_dtype(dummy.model.prediction_head, torch.float16, "prediction_head")
    assert all(param.dtype == torch.float16 for param in dummy.model.prediction_head.parameters())

    save_dir = tmp_path / "check"
    _safe_save_state_dict(dummy.model.language_model, str(save_dir), "language_model.bin", "dummy language model")
    saved_file = save_dir / "language_model.bin"
    assert saved_file.exists()
    state = torch.load(saved_file)
    assert any("lora_" in key for key in state.keys())
