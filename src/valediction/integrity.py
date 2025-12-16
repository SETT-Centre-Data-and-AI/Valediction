from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from re import Pattern
from typing import Any

from valediction.data_types.data_types import DataType
from valediction.support import list_as_bullets

ROOT = Path(__file__).resolve().parent
DIR_DICTIONARY = ROOT / "dictionary"
TEMPLATE_DATA_DICTIONARY_PATH = (
    DIR_DICTIONARY / "template" / "PROJECT - Data Dictionary.xltx"
)


externally_injected_variables: dict[
    str, Any
] = {}  # External injection store for package wrapping (any keys, always included)


def reset_injected_config_variables() -> None:
    global externally_injected_variables
    externally_injected_variables = {}


def inject_config_variables(variables: dict[str, Any]) -> None:
    """Injects variables into the Valediction Config, which will always be incorporated
    as overrides, regardless of Config calling method (default, session-scoped, or
    contextual).

    Args:
        variables (dict[str, Any]): Dictionary of config variables.
    """
    global externally_injected_variables, session_config

    # check type allows
    if not isinstance(variables, dict):
        raise TypeError(
            f"Config injection variables must be a dictionary, not {type(variables)}"
        )
    problematic_keys = []
    for variable_name in variables.keys():
        if not isinstance(variable_name, str):
            problematic_keys.append(variable_name)

    if problematic_keys:
        raise TypeError("Config injection variables accepts only string keys.")

    externally_injected_variables = dict(variables or {})

    # Apply immediately to the current session config (if it exists)
    if session_config is not None:
        _apply_external_injections(session_config)


def _apply_external_injections(config: Config) -> None:
    for variable_name, variable_value in externally_injected_variables.items():
        setattr(config, variable_name, deepcopy(variable_value))


class Config:
    def __init__(self):
        self.template_data_dictionary_path: Path = TEMPLATE_DATA_DICTIONARY_PATH
        self.max_table_name_length: int = 63
        self.max_column_name_length: int = 30
        self.max_primary_keys: int = 7
        self.invalid_name_pattern: str | Pattern = re.compile(r"[^A-Za-z0-9_]")
        self.null_values: list[str] = ["", "null", "none"]
        self.forbidden_characters: list[str] = []
        self.date_formats: dict[str, DataType] = {
            "%Y-%m-%d": DataType.DATE,
            "%Y/%m/%d": DataType.DATE,
            "%d/%m/%Y": DataType.DATE,
            "%d-%m-%Y": DataType.DATE,
            "%m/%d/%Y": DataType.DATE,
            "%m-%d-%Y": DataType.DATE,
            "%Y-%m-%d %H:%M:%S": DataType.DATETIME,
            "%Y-%m-%d %H:%M": DataType.DATETIME,
            "%d/%m/%Y %H:%M:%S": DataType.DATETIME,
            "%d/%m/%Y %H:%M": DataType.DATETIME,
            "%m/%d/%Y %H:%M:%S": DataType.DATETIME,
            "%Y-%m-%dT%H:%M:%S": DataType.DATETIME,
            "%Y-%m-%dT%H:%M:%S.%f": DataType.DATETIME,
            "%Y-%m-%dT%H:%M:%S%z": DataType.DATETIME,
            "%Y-%m-%dT%H:%M:%S.%f%z": DataType.DATETIME,
            "%Y-%m-%dT%H:%M:%SZ": DataType.DATETIME,
            "%Y-%m-%dT%H:%M:%S.%fZ": DataType.DATETIME,
        }
        self.enforce_no_null_columns: bool = True
        self.enforce_primary_keys: bool = True
        _apply_external_injections(self)

    def __repr__(self):
        date_list = list_as_bullets(
            elements=[f"{k}: {v.name} " for k, v in self.date_formats.items()],
            bullet="\n  - ",
        )
        return (
            f"Config(\n"
            f"Dictionary Settings:\n"
            f" - template_data_dictionary_path='{self.template_data_dictionary_path}'\n"
            f" - max_table_name_length={self.max_table_name_length}\n"
            f" - max_column_name_length={self.max_column_name_length}\n"
            f" - max_primary_keys={self.max_primary_keys}\n"
            f" - invalid_name_pattern={self.invalid_name_pattern}\n"
            f"Data Settings:\n"
            f" - default_null_values={self.null_values}\n"
            f" - forbidden_characters={self.forbidden_characters}\n"
            f" - date_formats=[{date_list}\n  ]\n"
            ")"
        )

    # Context Wrapper With Reset
    def __enter__(self):
        global session_config

        _apply_external_injections(self)

        session_config = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global session_config
        session_config = Config()


session_config: Config = None


def get_config() -> Config:
    """Gets the current `session_config` instance. Changing attributes will set them
    globally for the python session. Use `reset_default_config()` to reset to original
    defaults.

    Returns:
        Config: The current session configuration.
    """
    global session_config
    return session_config


def reset_default_config() -> None:
    """Resets `default_config` settings globally to original defaults."""
    global session_config
    session_config = Config()


reset_default_config()
