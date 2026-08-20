"""NUTQ speech recognition models."""

from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WhisperFeatureExtractor,
)

from .configuration_nutq import NutqConfig
from .modeling_nutq import NutqForConditionalGeneration, NutqModel
from .processing_nutq import NutqProcessor

__version__ = "0.1.0.dev0"

AutoConfig.register(NutqConfig.model_type, NutqConfig)
AutoModelForSpeechSeq2Seq.register(NutqConfig, NutqForConditionalGeneration)
AutoFeatureExtractor.register(NutqConfig, WhisperFeatureExtractor)
AutoProcessor.register(NutqConfig, NutqProcessor)

__all__ = [
    "NutqConfig",
    "NutqForConditionalGeneration",
    "NutqModel",
    "NutqProcessor",
    "__version__",
]
