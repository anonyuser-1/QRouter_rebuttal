import torch

from qrouter.models.vlms.losses import cis_patch_prior_loss, mask_to_patch_target


def test_mask_to_patch_target_uses_area_average():
    mask = torch.tensor(
        [
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ]
        ]
    )
    target = mask_to_patch_target(mask, patch_hw=(2, 2))
    expected = torch.tensor([[0.75, 0.0, 0.0, 1.0]])
    torch.testing.assert_close(target, expected)


def test_cis_loss_is_patch_prior_bce_plus_dice():
    target = torch.tensor([[1.0, 0.0, 0.5, 1.0]])
    perfect, perfect_parts = cis_patch_prior_loss(target, target)
    wrong, _ = cis_patch_prior_loss(1.0 - target, target)
    assert perfect < wrong
    assert set(perfect_parts) == {"cis_bce", "cis_dice"}
