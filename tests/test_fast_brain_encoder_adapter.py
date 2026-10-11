import pytest

from assistant.orchestration.fast_brain.encoder_adapter import (
    AdapterStatus, CAPABILITY_CLASSES, EncoderConfig, INTENT_CLASSES,
    MISSING_INFORMATION_CLASSES, MultilingualEncoderAdapter,
)


def test_missing_local_checkpoint_is_unavailable_without_download():
    adapter = MultilingualEncoderAdapter(
        EncoderConfig(local_model_path="this-path-does-not-exist"))
    assert adapter.status is AdapterStatus.UNAVAILABLE
    result = adapter.infer("open chrome")
    assert result.status is AdapterStatus.UNAVAILABLE
    assert result.intent_logits is None


def test_config_and_frozen_contract_mappings_are_deterministic():
    adapter = MultilingualEncoderAdapter(EncoderConfig(local_model_path=None))
    assert adapter.config.max_length == 160
    assert len(INTENT_CLASSES) == 9
    assert len(CAPABILITY_CLASSES) == 12
    assert len(MISSING_INFORMATION_CLASSES) == 10
    assert adapter.class_mappings["intent"] == INTENT_CLASSES


def test_masked_mean_pooling_excludes_padding():
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[[1.0, 3.0], [9.0, 9.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = MultilingualEncoderAdapter.masked_mean_pool(hidden, mask)
    assert torch.allclose(pooled, torch.tensor([[5.0, 6.0]]))


def test_masked_mean_pooling_handles_zero_mask():
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[[2.0, 4.0]]])
    mask = torch.tensor([[0]])
    pooled = MultilingualEncoderAdapter.masked_mean_pool(hidden, mask)
    assert torch.equal(pooled, torch.zeros((1, 2)))


def test_no_model_does_not_produce_decision_or_logits():
    adapter = MultilingualEncoderAdapter(EncoderConfig(local_model_path=None))
    output = adapter.infer("తెలుగు weather ఎలా ఉంది")
    assert output.status is AdapterStatus.UNAVAILABLE
    assert output.capability_logits is None

