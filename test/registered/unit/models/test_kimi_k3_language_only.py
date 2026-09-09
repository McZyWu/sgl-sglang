from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.models import kimi_k3
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "language_only,encoder_only", [(True, False), (False, False), (False, True)]
)
def test_language_only_skips_replicated_vision_parameters(language_only, encoder_only):
    config = SimpleNamespace(
        language_only=language_only,
        encoder_only=encoder_only,
        vision_config=object(),
        text_config=object(),
    )
    language = torch.nn.Linear(2, 2, bias=False)
    vision = torch.nn.Linear(2, 3, bias=False)
    projector = torch.nn.Linear(3, 2, bias=False)
    with (
        patch.object(kimi_k3, "KimiK3VisionTower", return_value=vision) as build_vision,
        patch.object(
            kimi_k3, "KimiK3MultiModalProjector", return_value=projector
        ) as build_projector,
        patch.object(
            kimi_k3, "KimiK3LinearForCausalLM", return_value=language
        ) as build_language,
    ):
        model = kimi_k3.KimiK3ForConditionalGeneration(config)
    assert build_vision.call_count == int(not language_only)
    assert build_projector.call_count == int(not language_only)
    assert build_language.call_count == int(not encoder_only)
    expected = (0 if encoder_only else 4) + (0 if language_only else 12)
    assert sum(p.numel() for p in model.parameters()) == expected
    if language_only:
        assert model.language_model is language
        assert model.vision_tower is None and model.mm_projector is None
        model.precompile_kernels_after_loading()
        with pytest.raises(ValueError, match="language-only"):
            model.get_image_feature([])
