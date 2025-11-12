import pytest

from valediction.data_types.data_types import DataType
from valediction.integrity import (
    Config,
    get_config,
    reset_default_config,
)


# Parameters
@pytest.fixture(autouse=True)
def _isolate_global_default():  # noqa
    """Ensure each test starts from a clean global default and leaves it clean."""
    reset_default_config()
    yield
    reset_default_config()


# Helpers
def _as_plain_dict(cfg: Config) -> dict:
    """Snapshot the config into plain-Python structures for easy comparison."""
    return {
        "template_data_dictionary_path": cfg.template_data_dictionary_path,
        "forbidden_characters": list(cfg.forbidden_characters),
        "max_table_name_length": cfg.max_table_name_length,
        "max_column_name_length": cfg.max_column_name_length,
        "max_primary_keys": cfg.max_primary_keys,
        "default_null_values": list(cfg.null_values),
        "invalid_name_pattern": getattr(
            cfg.invalid_name_pattern, "pattern", cfg.invalid_name_pattern
        ),
        "date_formats": dict(cfg.date_formats),
    }


# Tests
def test_defaults_match_fresh_Config():
    """Global default (after reset) equals a brand-new Config with original defaults."""
    reset_default_config()
    global_cfg = get_config()
    fresh_cfg = Config()

    assert _as_plain_dict(global_cfg) == _as_plain_dict(fresh_cfg)


def test_reset_default_config_restores_original_defaults():
    """After mutations, reset_default_config() brings the global default back to
    original values."""
    get_config().null_values.append("NULL")  # mutate nested field
    get_config().max_column_name_length = 999

    reset_default_config()
    restored = get_config()
    fresh = Config()
    assert _as_plain_dict(restored) == _as_plain_dict(fresh)


def test_date_formats_have_expected_entries_and_types():
    """date_formats should contain the expected keys and DataType values."""
    cfg = get_config()
    df = cfg.date_formats

    must_exist = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ]
    for k in must_exist:
        assert k in df, f"Missing expected format '{k}'"

    # Spot-check types
    assert df["%Y-%m-%d"] is DataType.DATE
    assert df["%Y-%m-%d %H:%M:%S"] is DataType.DATETIME


def test_mutating_nested_structures_affects_global_and_is_detected_against_fresh():
    cfg = get_config()

    # mutate nested list
    cfg.null_values.append("NA")
    assert "NA" in get_config().null_values

    # mutate nested dict
    cfg.date_formats["%d-%b-%Y"] = DataType.DATE
    assert get_config().date_formats["%d-%b-%Y"] is DataType.DATE

    # Fresh defaults remain baseline
    fresh = Config()
    assert "NA" not in fresh.null_values
    assert "%d-%b-%Y" not in fresh.date_formats


if __name__ == "__main__":
    pytest.main([__file__])
