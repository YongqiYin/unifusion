"""UniFusion: entropy-adaptive context assembly for heterogeneous RAG."""

from .baselines import dense_concat, naive_fusion
from .controller import ControllerConfig, UniFusionController
from .entropy import active_modalities, normalized_entropy
from .types import MODALITIES, ContextSelection, Corpus, CorpusItem, QueryScores

__version__ = "0.1.0"

__all__ = [
    "MODALITIES",
    "ContextSelection",
    "ControllerConfig",
    "Corpus",
    "CorpusItem",
    "QueryScores",
    "UniFusionController",
    "active_modalities",
    "dense_concat",
    "naive_fusion",
    "normalized_entropy",
    "__version__",
]
