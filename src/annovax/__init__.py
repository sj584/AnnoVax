"""AnnoVax: protective antigen prediction under positive-unlabeled learning.

Sequence (ESM-C 6B) and InterProScan annotation text (BioMistral-7B) embeddings are
concatenated and classified by an ensemble of 1,000 PU-bagged XGBoost models.
"""
__version__ = "1.0.0"
__all__ = ["bagging", "config", "data", "embed", "ensemble", "evaluate", "model",
           "preprocess", "report", "thresholds"]
