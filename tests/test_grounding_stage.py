import torch.nn as nn

from qrouter.models.grounding.qwen_sam import ConversationalGroundingModule


class TinySAM2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = nn.Linear(2, 2)
        self.sam_mask_decoder = nn.Linear(2, 2)


class TinyPromptAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(2, 2)

    def set_stage(self, stage: str) -> None:
        self.stage = stage


def make_grounding_module() -> ConversationalGroundingModule:
    module = ConversationalGroundingModule.__new__(ConversationalGroundingModule)
    nn.Module.__init__(module)
    module.sam_model = TinySAM2()
    module.prompt_adapter = TinyPromptAdapter()
    module.stage = "stage1"
    return module


def test_stage2_trains_sam2_mask_decoder_but_not_image_encoder():
    module = make_grounding_module()
    module.set_stage("stage2")
    module.train()

    assert all(not parameter.requires_grad for parameter in module.sam_model.image_encoder.parameters())
    assert all(parameter.requires_grad for parameter in module.sam_model.sam_mask_decoder.parameters())
    assert not module.sam_model.image_encoder.training
    assert module.sam_model.sam_mask_decoder.training
