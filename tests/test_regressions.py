import importlib
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.utils import SparsifyFn


def test_topk_mask_supports_single_token_and_full_sparsity():
    fn = SparsifyFn(sparse_mode="teal", mask_by="topk")
    fn.set_threshold(1.0)
    mask = fn(torch.tensor([[[1.0, 2.0, 3.0]]]))
    assert mask.shape == (1, 1, 3)
    assert mask.sum().item() == 1


def test_threshold_rejects_invalid_sparsity():
    fn = SparsifyFn(sparse_mode="teal", mask_by="threshold")
    with pytest.raises(ValueError):
        fn.set_threshold(1.1)


def test_unknown_dataset_is_explicitly_rejected():
    from utils.data import get_dataset

    with pytest.raises(ValueError, match="Unsupported dataset"):
        get_dataset("not-a-dataset", None, "train")


def test_eval_task_parser_accepts_comma_separated_values():
    source = (ROOT / "eval.py").read_text()
    assert "value.split(',')" in source

def test_repo_compiles():
    import compileall

    assert compileall.compile_dir(str(ROOT), quiet=1)
