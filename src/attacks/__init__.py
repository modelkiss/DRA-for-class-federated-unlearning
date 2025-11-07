"""Attack utilities exposed at the package level."""

from .label_inference import LabelInferenceResult, infer_forgotten_label
from .sensitive_feature_inference import SensitiveFeatureConfig, run_sensitive_feature_inference

__all__ = [
    "LabelInferenceResult",
    "infer_forgotten_label",
    "SensitiveFeatureConfig",
    "run_sensitive_feature_inference",
]
