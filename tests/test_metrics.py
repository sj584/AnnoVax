"""The label-free quantities that decide where the threshold goes."""
import numpy as np
import pytest

from annovax.metrics.pu_metrics import prior_threshold, pu_f1
from annovax.thresholds import THRESHOLD_RULES


def test_prior_threshold_predicts_the_declared_fraction():
    scores = np.linspace(0, 1, 100)
    for prior in (0.05, 0.25, 0.5):
        cut = prior_threshold(scores, prior)
        assert (scores >= cut).mean() == pytest.approx(prior, abs=0.01)


def test_prior_threshold_rejects_degenerate_priors():
    scores = np.linspace(0, 1, 10)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            prior_threshold(scores, bad)


def test_prior_threshold_is_scale_free():
    """It is a quantile rule, so a monotone rescaling must not move which rows fire."""
    scores = np.random.default_rng(0).random(200)
    fired = scores >= prior_threshold(scores, 0.25)
    rescaled = scores * 7 + 3
    assert np.array_equal(fired, rescaled >= prior_threshold(rescaled, 0.25))


def test_pu_f1_matches_its_definition():
    # 10 proteins, 4 labelled positive; the top 5 scores fire.
    observed = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1])
    threshold = 0.5
    recall = 3 / 4                      # 3 of the 4 labelled positives fire
    predicted_rate = 4 / 10             # 4 of 10 proteins fire
    assert pu_f1(observed, scores, threshold) == pytest.approx(recall ** 2 / predicted_rate)


def test_only_mcc_optimal_reads_labels():
    """The shipped threshold must come from a rule that never sees a label."""
    assert THRESHOLD_RULES["prior_transfer"] is False
    assert THRESHOLD_RULES["mcc_optimal"] is True
