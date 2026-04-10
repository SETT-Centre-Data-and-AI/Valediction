import importlib

import pytest

from valediction.data_types.data_types import DataType
from valediction.integrity import (
    Config,
    get_config,
    inject_config_variables,
    reset_default_config,
    reset_injected_config_variables,
)


# Parameters
@pytest.fixture(autouse=True)
def _isolate_global_default():  # noqa
    """Ensure each test starts clean and leaves it clean."""
    reset_injected_config_variables()
    reset_default_config()
    yield
    reset_injected_config_variables()
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
    config = get_config()
    df = config.date_formats

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
    assert df["%Y-%m-%d %H:%M:%S"] is DataType.TIMESTAMP


def test_mutating_nested_structures_affects_global_and_is_detected_against_fresh():
    config = get_config()

    # mutate nested list
    config.null_values.append("NA")
    assert "NA" in get_config().null_values

    # mutate nested dict
    config.date_formats["%d-%b-%Y"] = DataType.DATE
    assert get_config().date_formats["%d-%b-%Y"] is DataType.DATE

    # Fresh defaults remain baseline
    fresh = Config()
    assert "NA" not in fresh.null_values
    assert "%d-%b-%Y" not in fresh.date_formats


# Test Config Injection
def test_injection_applies_immediately_to_session_and_to_new_Config():
    reset_default_config()
    assert get_config().max_table_name_length == 63  # sanity check baseline

    inject_config_variables(
        {
            "max_table_name_length": 30,
            "test_variable": True,  # nonsense key should still attach
        }
    )

    # Applies immediately to the existing session_config
    config = get_config()
    assert config.max_table_name_length == 30
    assert config.test_variable is True

    # New Config() also picks it up (via __init__)
    fresh = Config()
    assert fresh.max_table_name_length == 30
    assert fresh.test_variable is True


def test_reset_default_config_keeps_injection():
    inject_config_variables({"max_table_name_length": 30})
    reset_default_config()

    assert get_config().max_table_name_length == 30
    assert Config().max_table_name_length == 30


def test_with_Config_context_applies_injection_even_if_instance_created_before_injection():
    config = Config()
    assert config.max_table_name_length == 63  # created before injection

    inject_config_variables({"max_table_name_length": 30})

    # __enter__ should apply injection to *this instance*
    with config as c:
        assert c.max_table_name_length == 30
        assert get_config().max_table_name_length == 30  # global now points to same

    # __exit__ calls reset_default_config(), which should still pick up injection
    assert get_config().max_table_name_length == 30


def test_importing_valediction_after_injection_does_not_clear_injection():
    inject_config_variables({"max_table_name_length": 30})

    # This is the scenario: Cynric imported & injected, later "import valediction"
    importlib.import_module("valediction")

    assert get_config().max_table_name_length == 30
    assert Config().max_table_name_length == 30


if __name__ == "__main__":
    pytest.main([__file__])
