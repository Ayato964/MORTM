"""
MORTM: Music Oriented Transformer Model (v5.0.0b2 Beta)
"""

__version__ = "5.0.0b2"

from .models.modules.config import (
    MORTMArgs,
    MORTM5Args,
    MORTM5_ABLATION_PRESETS,
    PAD_TOKEN_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    RVQ_TOKEN_OFFSET,
    INST_PROGRAM_OFFSET,
    INST_DRUMS_TOKEN_ID,
    INST_EMPTY_TOKEN_ID,
    NUM_SLOT_TOKENS,
    INST_RVQ_TOKEN_OFFSET,
    SLOT_CODE_DRUMS,
    SLOT_CODE_EMPTY,
    slot_code_to_token_id,
    token_id_to_slot_code,
)
from .models.mortm5 import (
    MidiVision,
    AblationTransformer1,
    AblationTransformer2,
    ABLATION_MODEL_CONFIGS,
    build_mortm5_ablation_model,
)
from .train.config import (
    TrainArgs,
    AbstractTrainSet,
    MORTM5_TRAIN_PRESETS,
)
from .train.muon import (
    HybridMuonOptimizer,
    build_muon_optimizer,
)
from .train.noam import (
    noam_lr,
    get_cosine_schedule_with_warmup,
)
from .train.datasets import (
    MORTM5MemmapDataset,
    MORTM5Collate,
    MORTM5OuterDataset,
    MORTM5WindowSubsetDataset,
    mortm5_collate_fn,
    mortm5_outer_collate_fn,
)
from .train.train import (
    MORTM5TrainSet,
    train_mortm5,
)

__all__ = [
    "__version__",
    "MORTMArgs",
    "MORTM5Args",
    "MORTM5_ABLATION_PRESETS",
    "PAD_TOKEN_ID",
    "BOS_TOKEN_ID",
    "EOS_TOKEN_ID",
    "RVQ_TOKEN_OFFSET",
    "INST_PROGRAM_OFFSET",
    "INST_DRUMS_TOKEN_ID",
    "INST_EMPTY_TOKEN_ID",
    "NUM_SLOT_TOKENS",
    "INST_RVQ_TOKEN_OFFSET",
    "SLOT_CODE_DRUMS",
    "SLOT_CODE_EMPTY",
    "slot_code_to_token_id",
    "token_id_to_slot_code",
    "MidiVision",
    "AblationTransformer1",
    "AblationTransformer2",
    "ABLATION_MODEL_CONFIGS",
    "build_mortm5_ablation_model",
    "TrainArgs",
    "AbstractTrainSet",
    "MORTM5_TRAIN_PRESETS",
    "HybridMuonOptimizer",
    "build_muon_optimizer",
    "noam_lr",
    "get_cosine_schedule_with_warmup",
    "MORTM5MemmapDataset",
    "MORTM5Collate",
    "MORTM5OuterDataset",
    "MORTM5WindowSubsetDataset",
    "mortm5_collate_fn",
    "mortm5_outer_collate_fn",
    "MORTM5TrainSet",
    "train_mortm5",
]
