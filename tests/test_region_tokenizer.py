import torch

from qrouter.models.vlms.region_tokenizer import (
    BackgroundPooler,
    RegionTokenizer,
    build_noisy_or_prior,
)


def test_noisy_or_matches_equation_and_is_not_patch_normalized():
    masks = torch.tensor(
        [
            [
                [1.0, 0.5, 0.0],
                [0.5, 0.5, 0.0],
            ]
        ]
    )
    scores = torch.tensor([[0.8, 0.5]])
    prior = build_noisy_or_prior(masks, scores)
    expected = 1.0 - (1.0 - scores[:, 0:1] * masks[:, 0]) * (1.0 - scores[:, 1:2] * masks[:, 1])
    torch.testing.assert_close(prior, expected)
    assert not torch.isclose(prior.sum(), torch.tensor(1.0))


def test_background_pooling_matches_equation_8_forward_value():
    patches = torch.tensor([[[1.0], [3.0], [10.0]]])
    prior = torch.tensor([[0.0, 0.5, 0.9]])
    selected = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    background = BackgroundPooler()(patches, prior, selected)
    expected = ((1.0 - 0.5) * 3.0 + (1.0 - 0.9) * 10.0) / (
        (1.0 - 0.5) + (1.0 - 0.9) + 1e-6
    )
    torch.testing.assert_close(background, torch.tensor([[expected]]))


def test_background_pooling_adds_epsilon_to_small_denominator():
    patches = torch.tensor([[[1.0], [2.0], [9.0]]])
    prior = torch.tensor([[0.9999997, 0.9999997, 0.2]])
    selected = torch.tensor([[0.0, 0.0, 1.0]])
    background = BackgroundPooler()(patches, prior, selected)
    weights = (1.0 - prior) * (1.0 - selected)
    expected = torch.einsum(
        "bn,bnd->bd",
        weights / (weights.sum(dim=-1, keepdim=True) + 1e-6),
        patches,
    )
    torch.testing.assert_close(background, expected)


def test_default_budget_decodes_32_candidates_and_forms_161_tokens():
    torch.manual_seed(7)
    batch_size = 2
    patches = torch.randn(batch_size, 16, 8)
    masks = torch.rand(batch_size, 32, 12, 12)
    scores = torch.full((batch_size, 32), 0.9)
    questions = torch.randn(batch_size, 6)
    tokenizer = RegionTokenizer(
        vision_dim=8,
        question_dim=6,
        num_region_tokens=32,
        num_context_tokens=128,
        routing_hidden_dim=12,
    )
    output = tokenizer(
        patch_tokens=patches,
        patch_hw=(4, 4),
        masks=masks,
        scores=scores,
        question_embedding=questions,
    )
    assert output.visual_tokens.shape == (batch_size, 161, 8)
    assert output.region_valid_mask.all()


def test_confidence_filter_reports_actual_valid_masks():
    patches = torch.randn(1, 9, 4)
    masks = torch.ones(1, 32, 6, 6)
    scores = torch.cat(
        [torch.full((1, 5), 0.8), torch.full((1, 27), 0.2)],
        dim=1,
    )
    tokenizer = RegionTokenizer(
        vision_dim=4,
        question_dim=3,
        num_region_tokens=32,
        num_context_tokens=4,
        routing_hidden_dim=5,
        score_threshold=0.5,
    )
    output = tokenizer(
        patch_tokens=patches,
        patch_hw=(3, 3),
        masks=masks,
        scores=scores,
        question_embedding=torch.randn(1, 3),
    )
    assert output.region_valid_mask.sum().item() == 5
    assert torch.count_nonzero(output.region_tokens[:, 5:]) == 0


def test_straight_through_topk_delivers_gradient_to_router():
    torch.manual_seed(11)
    tokenizer = RegionTokenizer(
        vision_dim=5,
        question_dim=4,
        num_region_tokens=3,
        num_context_tokens=2,
        routing_hidden_dim=7,
    )
    output = tokenizer(
        patch_tokens=torch.randn(2, 9, 5),
        patch_hw=(3, 3),
        masks=torch.rand(2, 3, 8, 8),
        scores=torch.full((2, 3), 0.9),
        question_embedding=torch.randn(2, 4),
    )
    loss = output.context_tokens.square().mean() + output.background_token.square().mean()
    loss.backward()
    gradient_norm = sum(
        float(parameter.grad.abs().sum())
        for parameter in tokenizer.routing_scorer.parameters()
        if parameter.grad is not None
    )
    assert gradient_norm > 0.0
