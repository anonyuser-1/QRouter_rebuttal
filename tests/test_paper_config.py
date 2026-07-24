from pathlib import Path

from qrouter.conf import load_config


def test_qrouter_b_paper_defaults_resolve_to_claimed_budgets():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "qrouter_b_paper.yaml"
    config = load_config(config_path)
    model = config["model"]
    optimization = config["optimization"]

    assert model["max_region_candidates"] == 32
    assert model["num_region_tokens"] == 32
    assert model["num_context_tokens"] == 128
    assert (model["num_region_tokens"] + model["num_context_tokens"] + 1) == 161

    world_size = 8
    resolved_global_batch = (
        optimization["per_device_batch_size"] * optimization["gradient_accumulation_steps"] * world_size
    )
    assert resolved_global_batch == optimization["global_batch_size"] == 128
    assert optimization["per_device_batch_size"] == 16
    assert optimization["gradient_accumulation_steps"] == 1
    assert optimization["distributed_strategy"] == "fsdp"
    assert optimization["stage1_steps"] == 20000
    assert optimization["stage2_steps"] == 10000
    assert optimization["learning_rate"] == 2.0e-5
    assert optimization["weight_decay"] == 0.1
    assert optimization["warmup_ratio"] == 0.03
    assert optimization["cis_dice_weight"] == 0.25
