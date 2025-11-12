import re
from typing import Any, Literal

from pandas import Series
from pandas import isna as _pd_isna

from valediction.data_types.data_types import DataType
from valediction.exceptions import DataDictionaryImportError
from valediction.integrity import get_config


def _check_name(name: str, entity: Literal["table", "column"]) -> list[str]:
    if entity not in ["table", "column"]:
        raise ValueError("entity must be either 'table' or 'column'")

    errors: list = []
    config = get_config()
    invalid_chars = (
        config.invalid_name_pattern
        if isinstance(config.invalid_name_pattern, re.Pattern)
        else re.compile(config.invalid_name_pattern)
    )
    max_name_length = (
        config.max_table_name_length
        if entity == "table"
        else config.max_column_name_length
    )

    if name != name.upper():  # name must be uppercase
        errors.append("must be uppercase")

    if invalid_chars.search(name):  # check invalid characters
        bad = set(invalid_chars.findall(name))
        errors.append(
            f"invalid characters: '{''.join(sorted(bad))}'; "
            "only A-Z, 0-9, and underscores are allowed with no whitespace"
        )

    if len(name) > max_name_length:  # max length 30
        errors.append(f"exceeds max length of {max_name_length}")

    if not name[0].isalpha():  # column starts with a letter
        errors.append("must start with a letter")

    if name.endswith("_"):  # column cannot end with an underscore
        errors.append("cannot end with '_'")

    if "__" in name:  # column cannot contain double underscores
        errors.append("cannot contain double underscores '__'")

    return errors


def _check_order(order: int | None) -> list[str]:
    errors: list = []
    if order is None:  # presence
        errors.append("order is required and must be an integer ≥ 1")
        return errors

    if not isinstance(order, int):  # type integer
        errors.append("order must be an integer ≥ 1")
        return errors

    if order < 1:  # must be ≥ 1
        errors.append("order must be ≥ 1")
        return errors

    return errors


def _check_data_type(data_type: DataType, length: int | None) -> list[str]:
    errors: list = []
    if not isinstance(data_type, DataType):  # Ensure is a DataType
        errors.append("data type is invalid; must be a DataType object")

    if length is not None:  # length rules
        if not isinstance(length, int):
            errors.append("length must be an positive integer if provided")
        if length <= 0:  # must be positive
            errors.append("length must be an positive integer if provided")

    if data_type == DataType.TEXT:  # required for DataType.TEXT
        if length is None:
            errors.append("length is required for TEXT columns")
    else:
        if length is not None:  # length not applicable
            errors.append(f"length is not applicable to {data_type.value} columns")

    return errors


def _check_primary_key(primary_key: int | None, data_type: DataType) -> list[str]:
    errors: list = []
    if primary_key is None:
        return errors

    if (
        not isinstance(primary_key, int)
        or primary_key < 1
        or primary_key > get_config().max_primary_keys
    ):
        errors.append(
            "primary key order must be an integer between 1 and 7 if provided"
        )

    if (
        hasattr(data_type, "valid_for_primary_key")
        and not data_type.valid_for_primary_key()
    ):
        errors.append(
            f"invalid data type '{data_type.value}' for primary key column; "
            "primary keys must be Text, Integer, Date, or Datetime"
        )

    return errors


def _normalise_name(name: str) -> str:
    return name.upper().strip()


def _norm_header_map(columns: list) -> dict:
    mapping, _ = {}, set()
    for c in columns:
        k = str(c).strip().lower().replace(" ", "_").replace("-", "_")
        if k in mapping:  # collision
            raise DataDictionaryImportError(
                f"Ambiguous headers after normalisation: {mapping[k]!r} and {c!r} both map to {k!r}"
            )
        mapping[k] = c
    return mapping


def _get_required_header(header_map: dict[str, str], key: str) -> str:
    if key not in header_map:
        raise DataDictionaryImportError(
            f"Required Data Dictionary column '{key}' not found. Available: {list(header_map.keys())}"
        )
    return header_map[key]


def _is_missing(val: Any) -> bool:
    return _pd_isna(val) or (isinstance(val, str) and val.strip() == "")


def _parse_truthy(val: Any) -> bool:
    if isinstance(val, str):
        return val.strip().lower() in {"y", "yes", "true", "1"}
    if isinstance(val, (int, float)):
        try:
            return int(val) == 1
        except Exception:
            return False
    return False


def _row_is_blank(row: Series, keys: tuple[str, str]) -> bool:
    a, b = keys
    return _is_missing(row[a]) and _is_missing(row[b])


def _parse_int(
    value: Any, label: str, row_ctx: str, *, required: bool = True
) -> int | None:
    if _is_missing(value):
        if required:
            raise DataDictionaryImportError(f"{row_ctx}: {label} is required.")
        return None
    try:
        return int(value)
    except Exception as e:
        raise DataDictionaryImportError(
            f"{row_ctx}: {label} must be integer (got {value!r})."
        ) from e
