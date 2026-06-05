from src.utils.config import load_config


def test_load_config():
    cfg = load_config()
    assert 'default' in cfg
