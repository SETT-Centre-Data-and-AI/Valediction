from __future__ import annotations

import re
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext
from numbers import Integral
from typing import List, Literal

from numpy import flatnonzero
from pandas import NA, DataFrame, Series, to_numeric
from pandas.api.types import is_bool_dtype, is_integer_dtype
from pandas.util import hash_pandas_object

from valediction.data_types.data_type_helpers import _DatetimeResolver
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
_INTEGER_TEXT = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_INTEGER_CACHE_LIMIT = 65_536


def _parse_integer_text(text: str, limit: Decimal) -> int | None:
    if _INTEGER_TEXT.fullmatch(text):
        try:
            numeric = Decimal(text)
            nearest = numeric.to_integral_value(rounding=ROUND_HALF_EVEN)
            if _INT64_MIN <= nearest <= _INT64_MAX:
                with localcontext() as context:
                    context.prec = max(28, len(numeric.as_tuple().digits) + 20)
                    if abs(numeric - nearest) <= limit:
                        return int(nearest)
        except InvalidOperation:
            pass
    return None


def _parse_integer(
    column: Series, *, errors: Literal["coerce", "raise"], tolerance: float = 1e-12
) -> Series:
    if is_integer_dtype(column.dtype) or is_bool_dtype(column.dtype):
        invalid = ((column < _INT64_MIN) | (column > _INT64_MAX)).fillna(False)
        parsed = column.mask(invalid, 0).astype("Int64").mask(invalid)
    else:
        converted = [NA] * len(column)
        values = column.array
        limit = Decimal(str(tolerance))
        cache: dict[str, int | None] = {}
        for position in flatnonzero(column.notna().to_numpy()):
            value = values[position]
            text = (
                str(int(value)) if isinstance(value, Integral) else str(value).strip()
            )
            if text not in cache:
                result = _parse_integer_text(text, limit)
                if len(cache) < _INTEGER_CACHE_LIMIT:
                    cache[text] = result
            else:
                result = cache[text]
            if result is not None:
                converted[position] = result
        parsed = Series(converted, index=column.index, name=column.name, dtype="Int64")
        invalid = column.notna() & parsed.isna()
    if errors == "raise" and invalid.any():
        raise ValueError(
            "Non-null integer values must be integer-equivalent and within signed 64-bit range"
        )
    return parsed


def invalid_mask_integer(column: Series, *, tolerance: float = 1e-12) -> Series:
    """True where a non-null value cannot be treated as an integer without losing non-
    zero remainder.

    Accepts scientific notation (e.g. '1e2').
    """
    parsed = _parse_integer(column, errors="coerce", tolerance=tolerance)
    return column.notna() & parsed.isna()


def invalid_mask_float(column: Series) -> Series:
    """True where non-null value is not convertible to a number."""
    notnull = column.notna()
    num = to_numeric(column, errors="coerce")
    return notnull & num.isna()


# Data Type Checks Date
def invalid_mask_date(column: Series, fmt: str | None) -> Series:
    """Must not contain a non-zero time component."""
    parsed = _DatetimeResolver(DataType.DATE, fmt).parse(column, errors="coerce")
    return column.notna() & parsed.isna()


def _parse_timestamp(
    column: Series, fmt: str | None, *, errors: Literal["coerce", "raise"]
) -> Series:
    return _DatetimeResolver(DataType.TIMESTAMP, fmt).parse(column, errors=errors)


def invalid_mask_datetime(column: Series, fmt: str | None) -> Series:
    notnull = column.notna()

    parsed = _parse_timestamp(column, fmt, errors="coerce")
    return notnull & parsed.isna()


# Other Text Checks
def invalid_mask_text_too_long(column: Series, max_len: int) -> Series:
    if max_len is None or max_len <= 0:
        # treat as unlimited length
        return Series(False, index=column.index)

    notnull = column.notna()
    s_txt = column.astype("string", copy=False)
    lens = s_txt.str.len()

    return notnull & (lens > max_len)


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
def apply_data_types(df: DataFrame, table_dictionary: Table) -> DataFrame:
    df = _set_nulls(df)
    # name -> column object
    column_dictionary = {_normalise(column.name): column for column in table_dictionary}

    for col in df.columns:
        data_type = column_dictionary.get(_normalise(col)).data_type
        datetime_format = column_dictionary.get(_normalise(col)).datetime_format

        if data_type in (DataType.TEXT, DataType.FILE):
            df[col] = df[col].astype("string")

        elif data_type == DataType.INTEGER:
            df[col] = _parse_integer(df[col], errors="raise")

        elif data_type == DataType.FLOAT:
            df[col] = df[col].map(float, na_action="ignore").astype("Float64")

        elif data_type == DataType.DATE:
            dtv = _DatetimeResolver(DataType.DATE, datetime_format).parse(
                df[col], errors="raise"
            )
            df[col] = dtv.dt.normalize()  # midnight

        elif data_type == DataType.TIMESTAMP:
            df[col] = _parse_timestamp(df[col], datetime_format, errors="raise")

        else:
            # Fallback: keep as string
            df[col] = df[col].astype("string")

    return df


# Bigint Checks
def invalid_mask_integer_out_of_range(
    series: Series,
    invalid_integer_mask: Series | None = None,
) -> Series:
    """
    Returns a boolean mask for values that:
      - are integer-like under Valediction's integer rules, AND
      - fall outside PostgreSQL INTEGER (int4) range.
    """

    parsed = _parse_integer(series, errors="coerce")
    out = ((parsed < -(2**31)) | (parsed > 2**31 - 1)).fillna(False)
    if invalid_integer_mask is not None:
        out &= ~invalid_integer_mask
    return out
