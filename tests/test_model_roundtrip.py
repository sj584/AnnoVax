"""Save -> load -> predict must reproduce the ensemble that trained.

This is the test that guards inference. The boosters are only useful if the scores they
give back after a reload are the ones the training run reported; a mismatch here would
be invisible in any metric and wrong in every prediction.
"""
import json

import numpy as np
import pytest

from annovax.bagging import run_bagging
from annovax.config import RunConfig
from annovax.model import AnnoVaxEnsemble, write_manifest

xgboost = pytest.importorskip("xgboost")


@pytest.fixture
def trained(tmp_path):
    """A 4-bag ensemble over 2560+4096 features, saved with its manifest."""
    rng = np.random.default_rng(0)
    dim = 2560 + 4096
    X_p = rng.standard_normal((20, dim)).astype(np.float32) + 0.5
    X_u = rng.standard_normal((60, dim)).astype(np.float32)
    X_test = rng.standard_normal((15, dim)).astype(np.float32)

    cfg = RunConfig(tag="test", k=4, ratio=2.0, p_ratio=0.8, gpus=(0,),
                    num_boost_round=5, out_root=tmp_path)
    object.__setattr__(cfg, "xgb_params",
                       lambda seed: {**RunConfig.xgb_params(cfg, seed), "device": "cpu"})

    model_dir = tmp_path / "model"
    bags = run_bagging(X_p, X_u, X_test, X_test[:5], cfg, tmp_path / "scratch",
                       save_dir=model_dir / "bags")
    write_manifest(model_dir, arm="esmc_biomistral", threshold=0.4,
                   headline_prior=0.25, n_bags=cfg.k, train_rows=80, train_positives=20)
    return model_dir, X_test, bags


def test_reloaded_scores_match_the_training_run(trained):
    model_dir, X_test, bags = trained
    ensemble = AnnoVaxEnsemble.load(model_dir)
    assert len(ensemble) == 4
    np.testing.assert_allclose(ensemble.predict(X_test), bags.scores_test.mean(axis=0),
                               rtol=0, atol=1e-6)


def test_return_bags_gives_the_per_bag_matrix(trained):
    model_dir, X_test, bags = trained
    mean, per_bag = AnnoVaxEnsemble.load(model_dir).predict(X_test, return_bags=True)
    assert per_bag.shape == bags.scores_test.shape
    np.testing.assert_allclose(mean, per_bag.mean(axis=0), atol=1e-7)
    # Bags are saved in index order, so each row must match its own bag.
    np.testing.assert_allclose(per_bag, bags.scores_test, atol=1e-6)


def test_manifest_records_the_feature_block_order(trained):
    manifest = AnnoVaxEnsemble.load(trained[0]).manifest
    assert [b["encoder"] for b in manifest["feature_blocks"]] == ["esmc_6b", "biomistral_7b"]
    assert manifest["feature_dim"] == 6656
    assert manifest["threshold_rule"] == "prior_transfer"


def test_predict_frame_applies_the_shipped_threshold(trained):
    model_dir, X_test, _ = trained
    ensemble = AnnoVaxEnsemble.load(model_dir)
    frame = ensemble.predict_frame([f"p{i}" for i in range(len(X_test))], X_test)
    assert list(frame.columns) == ["ID", "score_mean", "score_std", "pred"]
    assert frame["pred"].tolist() == (frame["score_mean"] >= ensemble.threshold).astype(int).tolist()


def test_wrong_feature_width_is_refused(trained):
    ensemble = AnnoVaxEnsemble.load(trained[0])
    with pytest.raises(ValueError, match="expected 6656 features"):
        ensemble.predict(np.zeros((3, 100), dtype=np.float32))


def test_limit_loads_a_prefix_of_the_bags(trained):
    model_dir, X_test, bags = trained
    ensemble = AnnoVaxEnsemble.load(model_dir, limit=2)
    assert len(ensemble) == 2
    np.testing.assert_allclose(ensemble.predict(X_test),
                               bags.scores_test[:2].mean(axis=0), atol=1e-6)


def test_missing_manifest_is_an_error_not_a_guess(trained):
    model_dir = trained[0]
    (model_dir / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="manifest.json is missing"):
        AnnoVaxEnsemble.load(model_dir)


def test_truncated_download_is_detected(trained):
    model_dir = trained[0]
    next(iter(sorted((model_dir / "bags").glob("bag_*.ubj")))).unlink()
    with pytest.raises(RuntimeError, match="manifest declares 4 bags but 3"):
        AnnoVaxEnsemble.load(model_dir)


def test_future_format_version_is_refused(trained):
    model_dir = trained[0]
    path = model_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["format_version"] = 99
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="format v99"):
        AnnoVaxEnsemble.load(model_dir)
