from __future__ import annotations

import torch
import torch.nn as nn

from qrouter.training.checkpointing import load_checkpoint, save_checkpoint


class TinyGroundingModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sam_model = nn.Linear(3, 3)
        self.prompt_adapter = nn.Linear(3, 3)
        for parameter in self.sam_model.parameters():
            parameter.requires_grad = False


class TinyQRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.grounding_module = TinyGroundingModule()
        self.router = nn.Linear(3, 3)


def test_checkpoint_schema_includes_sam2_and_loads_strictly(tmp_path):
    model = TinyQRouter()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=2e-5,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    target = tmp_path / "checkpoint.pt"
    expected_router = model.router.weight.detach().clone()
    expected_sam = model.grounding_module.sam_model.weight.detach().clone()

    save_checkpoint(
        path=target,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=10,
        stage="stage2",
        config={"experiment": {}, "model": {}, "data": {}, "optimization": {}, "evaluation": {}},
        epoch=4,
        metrics={"train_loss": 0.1},
        checkpoint_args={"stage": "stage2", "num_region_tokens": 32, "num_context_tokens": 128},
    )

    checkpoint = torch.load(target, map_location="cpu", weights_only=False)
    assert list(checkpoint) == [
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "step",
        "epoch",
        "args",
        "metrics",
    ]
    assert checkpoint["scaler"] == {}
    assert any(key.startswith("grounding_module.sam_model.") for key in checkpoint["model"])

    restored = TinyQRouter()
    load_checkpoint(target, restored, strict=True)
    torch.testing.assert_close(restored.router.weight, expected_router)
    torch.testing.assert_close(restored.grounding_module.sam_model.weight, expected_sam)
