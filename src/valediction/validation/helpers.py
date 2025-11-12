from __future__ import annotations

import re
from typing import List

from numpy import flatnonzero, round
from pandas import NA, DataFrame, Series, to_datetime, to_numeric
from pandas.util import hash_pandas_object

from valediction.data_types.data_types import DataType
from valediction.dictionary.model import Table
from valediction.integrity import get_config
from valediction.validation.issues import Range


# Remove Nulls
def _set_nulls(df: DataFrame) -> DataFrame:
    null_values = get_config().null_values
    token_set = {str(t).strip().casefold() for t in null_values}
    columns = df.select_dtypes(include=["string", "object"]).columns
    for column in columns:
        series = df[column]
        mask = series.notna() & series.str.casefold().isin(token_set)
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
    hash_col_name = "PK_HASH"
    if df_primaries.empty or df_primaries.shape[1] == 0:
        return Series([], dtype=object, name=hash_col_name)

    # Any NA in row => invalid PK -> None
    null_rows = df_primaries.isna().any(axis=1)

    # First Hash
    hash_1 = hash_pandas_object(df_primaries, index=False)  # uint64

    # Second Hash (rows backwards if single row, else salt)
    if df_primaries.shape[1] > 1:
        df_primaries_backwards = df_primaries.iloc[:, ::-1]
    else:
        s = df_primaries.iloc[:, 0]
        salt = Series(["§"] * len(s), index=s.index, dtype="string")
        df_primaries_backwards = DataFrame(
            {
                "_a": s,
                "_b": s.str.cat(salt),
            }
        )

    hash_2 = hash_pandas_object(df_primaries_backwards, index=False)  # uint64

    a1 = hash_1.to_numpy(dtype="uint64", copy=False).astype(object)
    a2 = hash_2.to_numpy(dtype="uint64", copy=False).astype(object)

    combined = (a1 << 64) | a2
    hashes = Series(
        combined, index=df_primaries.index, name=hash_col_name, dtype=object
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

    col_masks = df_primaries.apply(lambda s: s.str.contains(r"\s", na=False))

    return col_masks.any(axis=1)


# Data Type Checks Numeric
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
        vals = numeric[conversion_mask].astype("float64")
        frac = (vals - round(vals)).abs()
        invalid_conv = frac > tolerance
        invalid = invalid.copy()
        invalid.loc[conversion_mask] = invalid_conv.values
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


def _parse_ok_any(column: Series, formats: list[str]) -> Series:
    """
    Vectorised check: True for values that parse under at least one of `formats`.
    """
    if not formats:
        return Series(False, index=column.index)
    ok_any = Series(False, index=column.index)
    for fmt in formats:
        parsed = to_datetime(column, format=fmt, errors="coerce", utc=False)
        ok_any = ok_any | parsed.notna()
    return ok_any


def invalid_mask_date(column: Series, fmt: str | None) -> Series:
    """Must not contain a non-zero time component."""
    notnull = column.notna()

    if fmt:
        parsed = to_datetime(column, format=fmt, errors="coerce", utc=False)
        ok = parsed.notna()
        has_time = ok & (
            (parsed.dt.hour != 0)
            | (parsed.dt.minute != 0)
            | (parsed.dt.second != 0)
            | (parsed.dt.microsecond != 0)
        )
        return notnull & (~ok | has_time)

    allowed = _allowed_formats_for(DataType.DATE)
    ok_any = _parse_ok_any(column, allowed)
    return notnull & (~ok_any)


def invalid_mask_datetime(column: Series, fmt: str | None) -> Series:
    notnull = column.notna()

    if fmt:
        parsed = to_datetime(column, format=fmt, errors="coerce", utc=False)
        ok = parsed.notna()
        return notnull & (~ok)

    allowed = _allowed_formats_for(DataType.DATETIME)
    ok_any = _parse_ok_any(column, allowed)
    return notnull & (~ok_any)


# Other Text Checks
def invalid_mask_text_too_long(column: Series, max_len: int) -> Series:
    if max_len is None or max_len <= 0:
        # treat as unlimited length
        return Series(False, index=column.index)

    notnull = column.notna()
    lens = column.str.len()
    return notnull & (lens > max_len)


def invalid_mask_text_forbidden_characters(column: Series) -> Series:
    forbidden = get_config().forbidden_characters
    if not forbidden:
        return column.notna() & False

    pattern = "[" + re.escape("".join(forbidden)) + "]"
    notnull = column.notna()
    has_forbidden = column.str.contains(pattern, regex=True, na=False)
    return notnull & has_forbidden


# Apply Data Types #
def apply_data_types(df: DataFrame, table_dictionary: Table) -> DataFrame:
    # name -> column object
    column_dictionary = {column.name: column for column in table_dictionary}

    for col in df.columns:
        data_type = column_dictionary.get(col).data_type
        datetime_format = column_dictionary.get(col).datetime_format

        if data_type in (DataType.TEXT, DataType.FILE):
            df[col] = df[col].astype("string")

        elif data_type == DataType.INTEGER:
            # Accepts '12', '12.0', '1e2' etc.; validation guarantees integer-equivalent
            nums = to_numeric(df[col], errors="raise")
            df[col] = nums.round().astype("Int64")

        elif data_type == DataType.FLOAT:
            nums = to_numeric(df[col], errors="raise")
            df[col] = nums.astype("Float64")

        elif data_type == DataType.DATE:
            dtv = to_datetime(
                df[col], format=datetime_format, errors="raise", utc=False
            )
            df[col] = dtv.dt.normalize()  # midnight

        elif data_type == DataType.DATETIME:
            df[col] = to_datetime(
                df[col], format=datetime_format, errors="raise", utc=False
            )

        else:
            # Fallback: keep as string
            df[col] = df[col].astype("string")

    return df
