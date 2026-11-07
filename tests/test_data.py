"""EmbeddingRepo alignment: the join that silently ruins a model if it is wrong."""
import numpy as np
import pytest

from annovax.data import EmbeddingRepo


@pytest.fixture
def caches(tmp_path):
    """Two blocks whose rows are stored in deliberately different orders."""
    ids = ["p1", "p2", "p3"]
    a = tmp_path / "corpus__a.npz"
    b = tmp_path / "corpus__b.npz"
    np.savez(a, ids=np.array(ids), X=np.array([[1.0], [2.0], [3.0]], dtype=np.float32))
    np.savez(b, ids=np.array(ids[::-1]),
             X=np.array([[30.0], [20.0], [10.0]], dtype=np.float32))
    return a, b


def test_blocks_concatenate_in_argument_order(caches):
    repo = EmbeddingRepo(*caches)
    assert repo.dim == 2
    X, missing = repo.matrix(["p1", "p2"])
    assert missing == []
    # Second block is stored reversed; the lookup must undo that, not take row order.
    assert X.tolist() == [[1.0, 10.0], [2.0, 20.0]]


def test_rows_follow_the_requested_id_order(caches):
    X, _ = EmbeddingRepo(*caches).matrix(["p3", "p1"])
    assert X.tolist() == [[3.0, 30.0], [1.0, 10.0]]


def test_missing_ids_are_reported_not_filled(caches):
    X, missing = EmbeddingRepo(*caches).matrix(["p1", "ghost", "p2"])
    assert missing == ["ghost"]
    # A zero row would be scored confidently and mean nothing, so the ID is dropped
    # and named; callers turn that into an error.
    assert len(X) == 2


def test_id_present_in_only_one_block_counts_as_missing(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(a, ids=np.array(["p1", "p2"]), X=np.zeros((2, 1), dtype=np.float32))
    np.savez(b, ids=np.array(["p1"]), X=np.zeros((1, 1), dtype=np.float32))
    _, missing = EmbeddingRepo(a, b).matrix(["p1", "p2"])
    assert missing == ["p2"]


def test_output_is_float32_and_contiguous(caches):
    X, _ = EmbeddingRepo(*caches).matrix(["p1", "p2", "p3"])
    assert X.dtype == np.float32 and X.flags["C_CONTIGUOUS"]
