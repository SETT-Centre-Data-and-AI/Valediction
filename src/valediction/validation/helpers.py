from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, localcontext
from typing import List

import numpy as np
from numpy import flatnonzero
from pandas import NA, DataFrame, DatetimeTZDtype, Series, isna, to_datetime, to_numeric
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
)
from pandas.util import hash_pandas_object

from valediction.data_types.data_types import DataType
from valediction.dictionary.model import Table
from valediction.integrity import get_config
from valediction.support import _normalise
from valediction.validation.issues import Range


# Remove Nulls
def _set_nulls(df: DataFrame) -> DataFrame:
    null_values = get_config().null_values
    token_set = {str(t).strip().casefold() for t in null_values}
    columns = df.select_dtypes(include=["string", "object", "category"]).columns
    for column in columns:
        series = df[column]

        s_txt = series.astype("string", copy=False)  # dtype safe
        mask = s_txt.notna() & s_txt.str.strip().str.casefold().isin(token_set)
        if mask.any():
            df[column] = series.mask(mask, NA)

    return df


# Check for Nulls
def _column_has_values(column: Series):
    return column.notna().any()


# Range Setting
def mask_to_ranges(mask: Series, start_row: int) -> list[Range]:
    """Convert a boolean mask (over the current chunk) into 0-based contiguous
    ranges."""
    idx = flatnonzero(mask.to_numpy())
    if idx.size == 0:
        return []
    ranges: List[Range] = []
    run_start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        ranges.append(Range(start=start_row + run_start, end=start_row + prev))
        run_start = prev = i
    ranges.append(Range(start=start_row + run_start, end=start_row + prev))
    return ranges


# PK Hashes
def create_pk_hashes(
    df_primaries: DataFrame,
) -> Series:
    """For PK hash collision assessment, compute a deterministic 128-bit hash per row
    over the provided PK columns. This is created by computing two 64-bit hashes.

    forwards and backwards and then combining them. Rows with any NA across PK
    components are returned as None - flagging these for NULL violations.


    Args:
        df_primaries (DataFrame): DataFrame

    Returns:
        Series: Pandas Series with hashes or Nulls.
    """
    HASH_COL_NAME = "PK_HASH"
    if df_primaries.empty or df_primaries.shape[1] == 0:
        return Series([], dtype=object, name=HASH_COL_NAME)

    # Check Nulls
    null_rows = df_primaries.isna().any(axis=1)

    # Two independent 64-bit hashes with 16 byte keys
    hash_1 = hash_pandas_object(df_primaries, index=False, hash_key="valediction_pk1!")
    hash_2 = hash_pandas_object(df_primaries, index=False, hash_key="valediction_pk2!")

    # Combine into 128-bit integer keys
    a1 = hash_1.to_numpy(dtype="uint64", copy=False).astype(object)
    a2 = hash_2.to_numpy(dtype="uint64", copy=False).astype(object)
    combined = (a1 << 64) | a2

    hashes = Series(
        combined, index=df_primaries.index, name=HASH_COL_NAME, dtype=object
    )
    hashes[null_rows] = None
    return hashes


def compute_pk_masks(pk_hashes: Series, seen_hashes: set[int]) -> dict[str, Series]:
    """Compute masks for PK hashes that are either null or have been seen before.

    Args:
        pk_hashes (Series): Series of PK hashes.
        seen_hashes (set[int]): Set of hashes that have been seen before.

    Returns:
        dict[str, Series]: Dictionary for boolean masks:
        - null: rows where PK is None / NA
        - dup_full: rows that are part of a within-chunk duplicate group
        - cross_full: rows whose hash was seen in previous chunks (excluding dup_full)
        - new_first_full: rows that are the first occurrence of a hash
    """

    s = pk_hashes
    null = s.isna()
    valid = ~null
    if not valid.any():
        # empty/default masks
        return {
            "null": null,
            "in_chunk_collision": null,
            "cross_chunk_collision": null,
            "first_appearance": null,
        }

    s_valid = s[valid]

    # Within-chunk duplicate membership (mark *all* members)
    dup_local = s_valid.duplicated(keep=False)

    # Across-chunk duplicates (exclude those already in a local dup group)
    seen_local = s_valid.isin(seen_hashes)
    cross_local = seen_local & ~dup_local

    # New first occurrences in this chunk (first time we see the hash here, and not seen before)
    first_local = ~s_valid.duplicated(keep="first")
    new_first_local = first_local & ~seen_local

    # Lift back to full length masks
    in_chunk_collision = valid.copy()
    in_chunk_collision.loc[valid] = dup_local

    cross_chunk_collision = valid.copy()
    cross_chunk_collision.loc[valid] = cross_local

    first_appearance = valid.copy()
    first_appearance.loc[valid] = new_first_local

    return {
        "null": null,
        "in_chunk_collision": in_chunk_collision,
        "cross_chunk_collision": cross_chunk_collision,
        "first_appearance": first_appearance,
    }


# PK Whitespace
def pk_contains_whitespace_mask(df_primaries: DataFrame) -> Series:
    if df_primaries.empty or df_primaries.shape[1] == 0:
        return Series(False, index=df_primaries.index)

    col_masks = df_primaries.apply(
        lambda s: s.astype("string", copy=False).str.contains(r"\s", na=False)
    )
    return col_masks.any(axis=1)


# Data Type Checks Numeric
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_INT64_MIN_ABS = str(2**63)
_INT64_MAX_STR = str(_INT64_MAX)


def _invalid_decimal_integer(value: str, tolerance: Decimal) -> bool:
    try:
        number = Decimal(value)
    except InvalidOperation:
        return True

    if not number.is_finite():
        return True
    if number.adjusted() > 18:
        return True
    if abs(number) <= Decimal("0.5") and abs(number) <= tolerance:
        return False

    parts = number.as_tuple()
    with localcontext() as context:
        context.prec = max(
            context.prec,
            len(parts.digits) + max(0, -parts.exponent) + 2,
        )
        rounded = number.to_integral_value()
        fractional = abs(number - rounded)

    return fractional > tolerance or rounded < _INT64_MIN or rounded > _INT64_MAX


def _coerce_validated_integer(value: object) -> object:
    if isna(value):
        return NA
    return int(Decimal(str(value)).to_integral_value())


def invalid_mask_integer(column: Series, *, tolerance: float = 1e-12) -> Series:
    """True where a non-null value cannot be treated as an integer without losing non-
    zero remainder.

    Accepts scientific notation (e.g. '1e2').
    """
    notnull = column.notna()
    numeric = to_numeric(column, errors="coerce")
    invalid = notnull & numeric.isna()

    conversion_mask = notnull & numeric.notna()
    if conversion_mask.any():
        converted = numeric[conversion_mask]

        if is_integer_dtype(numeric.dtype):
            if numeric.dtype.kind == "u":
                invalid_conv = converted > _INT64_MAX
            else:
                invalid_conv = (converted < _INT64_MIN) | (converted > _INT64_MAX)
        elif is_float_dtype(column.dtype):
            values = converted.to_numpy(dtype="float64")
            finite = np.isfinite(values)
            rounded = np.rint(values)
            fractional = np.zeros(values.shape, dtype=bool)
            fractional[finite] = np.abs(values[finite] - rounded[finite]) > tolerance
            invalid_conv = Series(
                ~finite | fractional | (rounded < _INT64_MIN) | (rounded >= 2**63),
                index=converted.index,
            )
        elif is_bool_dtype(numeric.dtype):
            invalid_conv = Series(False, index=converted.index)
        else:
            text = column[conversion_mask].astype("string").str.strip()
            integer_token = text.str.fullmatch(r"[+-]?\d+", na=False)
            invalid_conv = Series(False, index=text.index)

            integer_values = text[integer_token]
            if not integer_values.empty:
                negative = integer_values.str.startswith("-")
                digits = integer_values.str.replace(r"^[+-]", "", regex=True)
                digits = digits.str.lstrip("0").mask(lambda values: values.eq(""), "0")
                limits = Series(
                    np.where(negative, _INT64_MIN_ABS, _INT64_MAX_STR),
                    index=digits.index,
                    dtype="string",
                )
                out_of_range = (digits.str.len() > limits.str.len()) | (
                    (digits.str.len() == limits.str.len()) & (digits > limits)
                )
                invalid_conv.loc[integer_values.index] = out_of_range

            decimal_values = text[~integer_token]
            if not decimal_values.empty:
                decimal_tolerance = Decimal(str(tolerance))
                invalid_conv.loc[decimal_values.index] = decimal_values.map(
                    lambda value: _invalid_decimal_integer(value, decimal_tolerance)
                )

        invalid.loc[conversion_mask] = invalid_conv.to_numpy(dtype=bool)
    return invalid


def invalid_mask_float(column: Series) -> Series:
    """True where non-null value is not convertible to a number."""
    notnull = column.notna()
    num = to_numeric(column, errors="coerce")
    return notnull & num.isna()


# Data Type Checks Date
def _allowed_formats_for(dtype: DataType) -> list[str]:
    """Return the list of formats from Config.date_formats allowed for the given
    DataType."""
    config = get_config()
    return [fmt for fmt, data_type in config.date_formats.items() if data_type == dtype]


def _date_order_for_format(fmt: str) -> str | None:
    day_position = fmt.find("%d")
    month_position = fmt.find("%m")
    year_position = fmt.find("%Y")
    if (
        day_position < 0
        or month_position < 0
        or (year_position >= 0 and year_position < min(day_position, month_position))
    ):
        return None
    return "dayfirst" if day_position < month_position else "monthfirst"


def _date_order_masks(
    column: Series, valid: Series, formats: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    supported_orders = {
        _date_order_for_format(fmt) for fmt in formats if fmt is not None
    }
    if not supported_orders.intersection({"dayfirst", "monthfirst"}):
        empty = np.zeros(len(column), dtype=bool)
        return empty, empty.copy()

    if is_datetime64_any_dtype(column.dtype):
        empty = np.zeros(len(column), dtype=bool)
        return empty, empty.copy()

    try:
        candidate_mask = column.str.match(
            r"^\s*\d{1,2}[/-]\d{1,2}[/-]\d{4}(?:\D|$)", na=False
        ).to_numpy(dtype=bool)
    except AttributeError:
        empty = np.zeros(len(column), dtype=bool)
        return empty, empty.copy()

    positions = np.flatnonzero(candidate_mask)
    if positions.size == 0:
        empty = np.zeros(len(column), dtype=bool)
        return empty, empty.copy()

    parts = column.iloc[positions].str.extract(
        r"^\s*(\d{1,2})[/-](\d{1,2})[/-]\d{4}(?:\D|$)", expand=True
    )
    first = to_numeric(parts[0], errors="coerce").to_numpy()
    second = to_numeric(parts[1], errors="coerce").to_numpy()
    is_valid = valid.to_numpy(dtype=bool)
    dayfirst = (
        (first > 12)
        & (second >= 1)
        & (second <= 12)
        & is_valid[positions]
        & ("dayfirst" in supported_orders)
    )
    monthfirst = (
        (second > 12)
        & (first >= 1)
        & (first <= 12)
        & is_valid[positions]
        & ("monthfirst" in supported_orders)
    )
    dayfirst_mask = np.zeros(len(column), dtype=bool)
    monthfirst_mask = np.zeros(len(column), dtype=bool)
    dayfirst_mask[positions] = dayfirst
    monthfirst_mask[positions] = monthfirst
    return dayfirst_mask, monthfirst_mask


def _prioritize_date_order(formats: list[str], order: str | None) -> list[str]:
    if order is None:
        return formats

    return sorted(
        formats,
        key=lambda fmt: (
            0
            if _date_order_for_format(fmt) == order
            else 1
            if _date_order_for_format(fmt) is None
            else 2
        ),
    )


def _parse_ok_any(column: Series, formats: list[str]) -> Series:
    notnull = column.notna()
    valid = Series(False, index=column.index)
    remaining = notnull.to_numpy()
    for fmt in formats:
        positions = remaining.nonzero()[0]
        if positions.size == 0:
            break
        parsed = to_datetime(
            column.iloc[positions], format=fmt, errors="coerce", utc=False
        )
        matched = parsed.notna().to_numpy()
        if matched.any():
            valid.iloc[positions[matched]] = True
            remaining[positions[matched]] = False
    return valid


def _parse_datetime_strict(column: Series, fmt: str) -> Series | None:
    try:
        return to_datetime(column, format=fmt, errors="raise", utc=False)
    except (TypeError, ValueError):
        return None


def _merge_datetime_matches(
    result: Series | None,
    parsed: Series,
    positions: np.ndarray,
    matched: np.ndarray,
    index,
) -> Series:
    if result is None:
        result = Series(index=index, dtype=parsed.dtype)
    elif result.dtype != parsed.dtype and (
        isinstance(result.dtype, DatetimeTZDtype)
        or isinstance(parsed.dtype, DatetimeTZDtype)
    ):
        result = result.astype(object)
    result.iloc[positions[matched]] = parsed.iloc[matched].array
    return result


def _parse_datetime_fallback(column: Series, formats: list[str]) -> Series:
    result = None
    remaining = column.notna().to_numpy()
    for fmt in formats:
        positions = remaining.nonzero()[0]
        if positions.size == 0:
            break
        parsed = to_datetime(
            column.iloc[positions], format=fmt, errors="coerce", utc=False
        )
        matched = parsed.notna().to_numpy()
        if matched.any():
            result = _merge_datetime_matches(
                result, parsed, positions, matched, column.index
            )
            remaining[positions[matched]] = False
    if result is None:
        return Series(index=column.index, name=column.name, dtype="datetime64[ns]")
    result.name = column.name
    return (
        to_datetime(result, errors="coerce", utc=False)
        if result.dtype == object
        else result
    )


def _to_datetime_with_formats(column: Series, formats: list[str]) -> Series:
    if is_datetime64_any_dtype(column.dtype):
        return column

    if formats:
        parsed = _parse_datetime_strict(column, formats[0])
        if parsed is not None:
            return parsed
    return _parse_datetime_fallback(column, formats)


def _is_midnight(parsed: Series) -> Series:
    return (
        (parsed.dt.hour == 0)
        & (parsed.dt.minute == 0)
        & (parsed.dt.second == 0)
        & (parsed.dt.microsecond == 0)
        & (parsed.dt.nanosecond == 0)
    )


def _date_parse_masks(parsed: Series) -> tuple[np.ndarray, np.ndarray]:
    matched = parsed.notna().to_numpy()
    if isinstance(parsed.dtype, DatetimeTZDtype):
        return matched, np.zeros(matched.shape, dtype=bool)
    if is_datetime64_any_dtype(parsed.dtype):
        valid = _is_midnight(parsed).fillna(False).to_numpy()
        return matched, matched & valid

    valid = np.zeros(matched.shape, dtype=bool)
    for position in np.flatnonzero(matched):
        value = parsed.iloc[position]
        timezone = getattr(value, "tzinfo", None)
        if timezone is not None and value.utcoffset() is not None:
            continue
        valid[position] = (
            getattr(value, "hour", 0) == 0
            and getattr(value, "minute", 0) == 0
            and getattr(value, "second", 0) == 0
            and getattr(value, "microsecond", 0) == 0
            and getattr(value, "nanosecond", 0) == 0
        )
    return matched, matched & valid


def _valid_date_fallback(column: Series, formats: list[str]) -> Series:
    valid = Series(False, index=column.index)
    remaining = column.notna().to_numpy()
    for fmt in formats:
        positions = remaining.nonzero()[0]
        if positions.size == 0:
            break
        parsed = to_datetime(
            column.iloc[positions], format=fmt, errors="coerce", utc=False
        )
        matched, accepted = _date_parse_masks(parsed)
        if accepted.any():
            valid.iloc[positions[accepted]] = True
        remaining[positions[matched]] = False
    return valid


def _invalid_date_mask(column: Series, parsed: Series | None = None) -> Series:
    notnull = column.notna()
    if isinstance(column.dtype, DatetimeTZDtype):
        return notnull
    values = column if parsed is None else parsed
    return notnull & ~_is_midnight(values)


def invalid_mask_date(column: Series, fmt: str | None) -> Series:
    """Reject non-midnight and timezone-aware values for DATE columns."""
    notnull = column.notna()
    formats = [fmt] if fmt else []
    formats.extend(
        candidate
        for candidate in (
            *_allowed_formats_for(DataType.DATE),
            *_allowed_formats_for(DataType.TIMESTAMP),
        )
        if candidate not in formats
    )
    if is_datetime64_any_dtype(column.dtype):
        return _invalid_date_mask(column)
    if formats:
        parsed = _parse_datetime_strict(column, formats[0])
        if parsed is not None:
            if isinstance(parsed.dtype, DatetimeTZDtype):
                return notnull
            _, accepted = _date_parse_masks(parsed)
            return notnull & ~Series(accepted, index=column.index)
    valid = _valid_date_fallback(column, formats)
    return notnull & ~valid


def invalid_mask_datetime(column: Series, fmt: str | None) -> Series:
    notnull = column.notna()
    if is_datetime64_any_dtype(column.dtype):
        return Series(False, index=column.index)

    allowed = [fmt] if fmt else []
    allowed.extend(
        candidate
        for candidate in _allowed_formats_for(DataType.TIMESTAMP)
        if candidate not in allowed
    )
    if allowed and _parse_datetime_strict(column, allowed[0]) is not None:
        return Series(False, index=column.index)
    ok_any = _parse_ok_any(column, allowed)
    return notnull & (~ok_any)


# Other Text Checks
def invalid_mask_text_too_long(column: Series, max_len: int) -> Series:
    if max_len is None or max_len <= 0:
        # treat as unlimited length
        return Series(False, index=column.index)

    notnull = column.notna()
    s_txt = column.astype("string", copy=False)
    lens = s_txt.str.len()

    return notnull & (lens > max_len)


def _primary_key_text_columns_over_length_limit(
    table: Table,
    primary_keys: list[str],
    max_length: int,
) -> list[str]:
    primary_key_names = {_normalise(name) for name in primary_keys}
    return [
        column.name
        for column in table
        if _normalise(column.name) in primary_key_names
        and column.data_type is DataType.TEXT
        and (column.length is None or column.length > max_length)
    ]


def invalid_mask_text_forbidden_characters(column: Series) -> Series:
    forbidden = get_config().forbidden_characters
    if not forbidden:
        return column.notna() & False

    pattern = "[" + re.escape("".join([str(s) for s in forbidden])) + "]"
    notnull = column.notna()

    s_txt = column.astype("string", copy=False)
    has_forbidden = s_txt.str.contains(pattern, regex=True, na=False)

    return notnull & has_forbidden


# Apply Data Types #
def apply_data_types(
    df: DataFrame,
    table_dictionary: Table,
    datetime_formats: dict[str, str | None] | None = None,
    datetime_orders: dict[str, str] | None = None,
) -> DataFrame:
    # name -> column object
    column_dictionary = {_normalise(column.name): column for column in table_dictionary}
    datetime_formats = datetime_formats or {}
    datetime_orders = datetime_orders or {}

    for col in df.columns:
        data_type = column_dictionary.get(_normalise(col)).data_type
        column = column_dictionary.get(_normalise(col))
        datetime_format = datetime_formats.get(column.name) or column.datetime_format
        datetime_order = datetime_orders.get(_normalise(column.name))

        if data_type in (DataType.TEXT, DataType.FILE):
            df[col] = df[col].astype("string")

        elif data_type == DataType.INTEGER:
            # Accepts '12', '12.0', '1e2' etc.; validation guarantees integer-equivalent
            source = df[col]
            nums = to_numeric(source, errors="raise")
            if is_integer_dtype(nums.dtype):
                df[col] = nums.astype("Int64")
            elif is_float_dtype(source.dtype):
                df[col] = nums.round().astype("Int64")
            else:
                df[col] = source.map(_coerce_validated_integer).astype("Int64")

        elif data_type == DataType.FLOAT:
            nums = to_numeric(df[col], errors="raise")
            df[col] = nums.astype("Float64")

        elif data_type == DataType.DATE:
            formats = [datetime_format] if datetime_format else []
            formats.extend(
                fmt
                for fmt in (
                    *_allowed_formats_for(DataType.DATE),
                    *_allowed_formats_for(DataType.TIMESTAMP),
                )
                if fmt not in formats
            )
            formats = _prioritize_date_order(formats, datetime_order)
            dtv = _to_datetime_with_formats(df[col], formats)
            df[col] = dtv.dt.normalize()  # midnight

        elif data_type == DataType.TIMESTAMP:
            formats = [datetime_format] if datetime_format else []
            formats.extend(
                fmt
                for fmt in _allowed_formats_for(DataType.TIMESTAMP)
                if fmt not in formats
            )
            formats = _prioritize_date_order(formats, datetime_order)
            df[col] = _to_datetime_with_formats(df[col], formats)

        else:
            # Fallback: keep as string
            df[col] = df[col].astype("string")

    return df


# Bigint Checks
_PG_INT4_MIN = -(2**31)
_PG_INT4_MAX = 2**31 - 1


def invalid_mask_integer_out_of_range(
    series: Series,
    invalid_integer_mask: Series | None = None,
) -> Series:
    """
    Returns a boolean mask for values that:
      - are integer-like under Valediction's integer rules, AND
      - fall outside PostgreSQL INTEGER (int4) range.
    """

    # Start with all-False mask
    out = series.isna() & False

    # Use caller-provided invalid mask to avoid recomputing if available
    if invalid_integer_mask is None:
        from valediction.validation.helpers import invalid_mask_integer  # avoid cycles

        invalid_integer_mask = invalid_mask_integer(series)

    # We only check range for values that already pass integer validation
    valid = (~invalid_integer_mask) & series.notna()
    if not valid.any():
        return out

    numeric = to_numeric(series[valid], errors="raise").round()
    out.loc[valid] = ((numeric < _PG_INT4_MIN) | (numeric > _PG_INT4_MAX)).to_numpy()
    return out
