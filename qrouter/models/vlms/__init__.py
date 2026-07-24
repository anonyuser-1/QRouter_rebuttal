from qrouter.models.vlms.losses import cis_patch_prior_loss, mask_to_patch_target
from qrouter.models.vlms.region_tokenizer import (
    RegionTokenizer,
    RegionTokenizerOutput,
)

__all__ = [
    "QRouterModel",
    "QRouterOutput",
    "RegionTokenizer",
    "RegionTokenizerOutput",
    "cis_patch_prior_loss",
    "mask_to_patch_target",
]


def __getattr__(name):
    if name in {"QRouterModel", "QRouterOutput"}:
        from qrouter.models.vlms.qrouter import QRouterModel, QRouterOutput

        return {"QRouterModel": QRouterModel, "QRouterOutput": QRouterOutput}[name]
    raise AttributeError(name)
