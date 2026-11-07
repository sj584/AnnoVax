"""PU bagging mechanics. Small K, CPU, synthetic features -- shape and determinism only."""
import numpy as np
import pytest

from annovax.bagging import bag_rng, run_bagging
from annovax.config import RunConfig

xgboost = pytest.importorskip("xgboost")


@pytest.fixture
def cfg(tmp_path):
    # device="cpu" so the suite runs without a GPU; everything else is the real config.
    config = RunConfig(tag="test", k=4, ratio=2.0, p_ratio=0.8, gpus=(0,),
                       num_boost_round=5, out_root=tmp_path)
    object.__setattr__(config, "xgb_params",
                       lambda seed: {**RunConfig.xgb_params(config, seed), "device": "cpu"})
    return config


@pytest.fixture
def arrays():
    rng = np.random.default_rng(0)
    return (rng.standard_normal((20, 8)).astype(np.float32),   # P
            rng.standard_normal((60, 8)).astype(np.float32),   # U
            rng.standard_normal((15, 8)).astype(np.float32),   # test
            rng.standard_normal((10, 8)).astype(np.float32))   # vaxign


def test_bag_rng_is_reproducible_and_bag_specific():
    assert bag_rng(0, 3).random() == bag_rng(0, 3).random()
    assert bag_rng(0, 3).random() != bag_rng(0, 4).random()
    # Seeded on the pair, not seed + k: additive seeds give correlated streams.
    assert bag_rng(1, 3).random() != bag_rng(0, 4).random()


def test_score_matrices_have_one_row_per_bag(cfg, arrays, tmp_path):
    result = run_bagging(*arrays, cfg, tmp_path / "scratch")
    assert result.scores_p.shape == (cfg.k, 20)
    assert result.scores_u.shape == (cfg.k, 60)
    assert result.scores_test.shape == (cfg.k, 15)
    assert result.scores_vaxign.shape == (cfg.k, 10)
    assert np.all((result.scores_test >= 0) & (result.scores_test <= 1))


def test_oob_masks_match_the_drawn_bag_sizes(cfg, arrays, tmp_path):
    result = run_bagging(*arrays, cfg, tmp_path / "scratch")
    n_psub = round(cfg.p_ratio * 20)                 # 16 positives per bag
    n_sub = round(cfg.ratio * n_psub)                # 32 unlabeled per bag
    assert np.all(result.oob_mask_p.sum(axis=1) == 20 - n_psub)
    assert np.all(result.oob_mask.sum(axis=1) == 60 - n_sub)
    # p_ratio < 1 is what gives the positives an out-of-bag score at all.
    assert np.isfinite(result.oob_stats_p()[0]).any()


def test_same_seed_reproduces_the_scores(cfg, arrays, tmp_path):
    first = run_bagging(*arrays, cfg, tmp_path / "a")
    second = run_bagging(*arrays, cfg, tmp_path / "b")
    np.testing.assert_allclose(first.scores_test, second.scores_test)


def test_bag_larger_than_the_pool_is_refused(cfg, arrays, tmp_path):
    object.__setattr__(cfg, "ratio", 100.0)
    with pytest.raises(ValueError, match="exceeds the unlabeled pool"):
        run_bagging(*arrays, cfg, tmp_path / "scratch")


def test_save_dir_writes_one_booster_per_bag(cfg, arrays, tmp_path):
    bags_dir = tmp_path / "model" / "bags"
    run_bagging(*arrays, cfg, tmp_path / "scratch", save_dir=bags_dir)
    assert len(sorted(bags_dir.glob("bag_*.ubj"))) == cfg.k
