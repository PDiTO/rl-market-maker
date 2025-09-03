from pathlib import Path

import pytest

from rlmm.config import load_config
from rlmm.env.features import DIRECTIONAL

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_shipped_configs_load() -> None:
    ppo = load_config(CONFIGS / "ppo.toml")
    assert ppo.env.n_steps == 600
    assert ppo.ppo.batch_size == 16 * 512
    assert ppo.train.seeds == (1, 2, 3)
    ablation = load_config(CONFIGS / "ppo_ablation.toml")
    assert ablation.env.hide_features == DIRECTIONAL
    assert ablation.env.replace(hide_features=()) == ppo.env
    ev = load_config(CONFIGS / "eval.toml")
    assert ev.env == ppo.env
    assert "ppo_seed1" in ev.eval.policies


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[ppo]\nlearning_rat = 1e-3\n")
    with pytest.raises(KeyError):
        load_config(bad)
    bad.write_text("[env.market]\nvolatilty = 2.0\n")
    with pytest.raises(KeyError):
        load_config(bad)
