import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from analysis.flexivit_config import (
    get_pretrained_metadata,
    inspect_flexivit_architectures,
    inspect_deit_architectures,
    generate_markdown_report,
)


def test_pretrained_metadata():
    meta = get_pretrained_metadata()
    assert "efficientnet_b3" in meta
    assert meta["efficientnet_b3"]["test_input_size"] == (3, 320, 320)
    assert meta["efficientnet_b3"]["input_size"] == (3, 288, 288)

    assert "flexivit_base" in meta
    assert meta["flexivit_base"]["input_size"] == (3, 240, 240)


def test_flexivit_architecture_inspection():
    flex = inspect_flexivit_architectures()
    # Check F-p
    assert flex["F-p"][224]["patch_size"] == (16, 16)
    assert flex["F-p"][224]["grid_size"] == (14, 14)
    assert flex["F-p"][448]["grid_size"] == (28, 28)
    assert flex["F-p"][448]["num_tokens"] == 784

    # Check F-t
    assert flex["F-t"][224]["patch_size"] == (14, 14)
    assert flex["F-t"][224]["grid_size"] == (16, 16)
    assert flex["F-t"][448]["patch_size"] == (28, 28)
    assert flex["F-t"][448]["grid_size"] == (16, 16)
    assert flex["F-t"][448]["num_tokens"] == 256


def test_model_configs_report_generation():
    meta = get_pretrained_metadata()
    flex = inspect_flexivit_architectures()
    deit = inspect_deit_architectures()
    md = generate_markdown_report(meta, flex, deit)
    assert "# Model Pretrained Configurations & Architecture Details" in md
    assert "240x240" in md
    assert "15x15 = 225 tokens" in md
