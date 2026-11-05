"""
MORTM: Music Oriented Transformer Model (v5.0.0b5 Beta)
"""

__version__ = "5.0.0b5"

from .models.modules.config import (
    MORTMArgs,
    MORTM5Args,
    MORTM5_ABLATION_PRESETS,
    MORTM5_SYMIC_MAIN_PRESETS,
    ABLATION_SUBDIR_MAP,
    resolve_ablation_subdir,
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
    OCTUPLE_DEFAULT_ATTR_SIZES,
    REMI_DEFAULT_ATTR_SIZES,
    slot_code_to_token_id,
    token_id_to_slot_code,
)
from .models.modules.layers import (
    DecodeCodebookHead,
    DecodeOctupleHead,
)
from .models.mortm5 import (
    MidiVision,
    AblationTransformer1,
    AblationTransformer2,
    OctupleTransformer,
    REMITransformer,
    ABLATION_MODEL_CONFIGS,
    SYMIC_MAIN_MODEL_CONFIGS,
    MODEL_MODE_REGISTRY,
    register_mortm5_model_mode,
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
    MORTM5SymbolicMemmapDataset,
    MORTM5Collate,
    MORTM5OuterDataset,
    MORTM5WindowSubsetDataset,
    PackedOptimizerStep,
    DynamicStepBatchPacker,
    mortm5_collate_fn,
    mortm5_outer_collate_fn,
)
from .train.train import (
    MORTM5TrainSet,
    train_mortm5,
)
from .utils.symbolic_tokenizers import (
    SegmentWindowSpec,
    SymbolicGridSpec,
    OctupleVocabSpec,
    REMIVocabSpec,
    DEFAULT_OCTUPLE_SPEC,
    DEFAULT_REMI_SPEC,
    AbstractSymbolicSegmentTokenizer,
    OctupleMMTTokenizer,
    MultitrackREMITokenizer,
    SYMBOLIC_TOKENIZER_REGISTRY,
    register_symbolic_tokenizer,
    build_symbolic_tokenizer,
)

_LAZY_INFER_EXPORTS = {
    "Mortm5SamplingConfig",
    "Mortm5PromptEncoder",
    "Mortm5MidiGenerator",
    "parse_slot_spec",
}


def __getattr__(name: str):
    if name in _LAZY_INFER_EXPORTS:
        from .utils import mortm5_infer

        return getattr(mortm5_infer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "MORTMArgs",
    "MORTM5Args",
    "MORTM5_ABLATION_PRESETS",
    "MORTM5_SYMIC_MAIN_PRESETS",
    "ABLATION_SUBDIR_MAP",
    "resolve_ablation_subdir",
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
    "OCTUPLE_DEFAULT_ATTR_SIZES",
    "REMI_DEFAULT_ATTR_SIZES",
    "slot_code_to_token_id",
    "token_id_to_slot_code",
    "DecodeCodebookHead",
    "DecodeOctupleHead",
    "MidiVision",
    "AblationTransformer1",
    "AblationTransformer2",
    "OctupleTransformer",
    "REMITransformer",
    "ABLATION_MODEL_CONFIGS",
    "SYMIC_MAIN_MODEL_CONFIGS",
    "MODEL_MODE_REGISTRY",
    "register_mortm5_model_mode",
    "build_mortm5_ablation_model",
    "TrainArgs",
    "AbstractTrainSet",
    "MORTM5_TRAIN_PRESETS",
    "HybridMuonOptimizer",
    "build_muon_optimizer",
    "noam_lr",
    "get_cosine_schedule_with_warmup",
    "MORTM5MemmapDataset",
    "MORTM5SymbolicMemmapDataset",
    "MORTM5Collate",
    "MORTM5OuterDataset",
    "MORTM5WindowSubsetDataset",
    "PackedOptimizerStep",
    "DynamicStepBatchPacker",
    "mortm5_collate_fn",
    "mortm5_outer_collate_fn",
    "MORTM5TrainSet",
    "train_mortm5",
    "SegmentWindowSpec",
    "SymbolicGridSpec",
    "OctupleVocabSpec",
    "REMIVocabSpec",
    "DEFAULT_OCTUPLE_SPEC",
    "DEFAULT_REMI_SPEC",
    "AbstractSymbolicSegmentTokenizer",
    "OctupleMMTTokenizer",
    "MultitrackREMITokenizer",
    "SYMBOLIC_TOKENIZER_REGISTRY",
    "register_symbolic_tokenizer",
    "build_symbolic_tokenizer",
    "Mortm5SamplingConfig",
    "Mortm5PromptEncoder",
    "Mortm5MidiGenerator",
    "parse_slot_spec",
]
