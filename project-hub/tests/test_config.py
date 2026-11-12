import pytest
from hub.config import ConfigError, find_root, load_config


def test_load_config_merges_local(hub_root, monkeypatch):
    monkeypatch.delenv("HUB_ME", raising=False)
    cfg = load_config(hub_root)
    assert cfg.name == "demo"
    assert set(cfg.people) == {"alice", "bob"}
    assert cfg.me == "alice"
    assert cfg.roots["CODE"] == "/here/code"


def test_env_overrides_me(hub_root, monkeypatch):
    monkeypatch.setenv("HUB_ME", "bob")
    assert load_config(hub_root).me == "bob"


def test_unknown_me_rejected(hub_root, monkeypatch):
    monkeypatch.setenv("HUB_ME", "carol")
    with pytest.raises(ConfigError, match="carol"):
        load_config(hub_root)


def test_missing_key_rejected(tmp_path):
    (tmp_path / "project.yaml").write_text("name: x\npeople: {}\n")
    with pytest.raises(ConfigError, match="series"):
        load_config(tmp_path)


def test_find_root_walks_up(hub_root):
    deep = hub_root / "Status" / "Items"
    deep.mkdir(parents=True)
    assert find_root(deep) == hub_root.resolve()


def test_find_root_fails(tmp_path):
    with pytest.raises(ConfigError):
        find_root(tmp_path)
