import re
from pathlib import Path
from re import Pattern

from valediction.data_types.data_types import DataType
from valediction.support import list_as_bullets

ROOT = Path(__file__).resolve().parent
DIR_DICTIONARY = ROOT / "dictionary"
TEMPLATE_DATA_DICTIONARY_PATH = (
    DIR_DICTIONARY / "template" / "PROJECT - Data Dictionary.xltx"
)


class Config:
    def __init__(self):
        self.template_data_dictionary_path: Path = TEMPLATE_DATA_DICTIONARY_PATH
        self.max_table_name_length: int = 63
        self.max_column_name_length: int = 30
        self.max_primary_keys: int = 7
        self.invalid_name_pattern: str | Pattern = re.compile(r"[^A-Z0-9_]")
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
        global default_config
        default_config = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global default_config
        default_config = Config()


default_config: Config = None


def get_config() -> Config:
    """Gets the current `default_config` instance. Changing attributes will set them
    globally.

    Returns:
        Config: The current default configuration.
    """
    global default_config
    return default_config


def reset_default_config() -> None:
    """Resets `default_config` settings globally to original defaults."""
    global default_config
    default_config = Config()


reset_default_config()
