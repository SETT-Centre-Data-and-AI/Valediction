from pandas import NA, Series, to_datetime

from valediction.data_types.data_types import DataType
from valediction.integrity import get_config


def infer_datetime_format(
    series: Series,
    slice_sample_size: int = 100,
) -> str | None:
    """Efficiently infers date/datetime format (or rules out) by looping over slices of
    the column series, ruling out formats.

    Args:
        series (Series): Column data series
        slice_sample_size (int, optional): Number of rows to test in each slice
            loop. Defaults to 100.

    Raises:
        ValueError: No values, or ambiguous format after full scan.

    Returns:
        str | None: datetime format string, or None if no format matches.
    """
    datetime_formats = get_config().date_formats.keys()
    values = series.str.strip().replace("", NA).dropna()
    if values.empty:
        raise ValueError("Series has no non-null values to test.")

    start_i = 0
    total = len(values)
    last_ambiguous: list[str] | None = None
    remaining = list(datetime_formats)

    # loop over slices
    while start_i < total:
        end_i = min(start_i + slice_sample_size, total)
        sample = values.iloc[start_i:end_i]

        valid_formats: list[str] = []
        for fmt in remaining:
            try:
                to_datetime(sample, format=fmt, errors="raise")
                valid_formats.append(fmt)
            except Exception:
                pass

        remaining = valid_formats

        # Decision
        current = valid_formats
        if len(current) == 1:
            return current[0]
        elif len(current) == 0:
            return None
        else:
            last_ambiguous = current
            start_i = end_i  # advance to next slice

    # all values scanned and format still ambiguous
    raise ValueError(f"Ambiguous datetime format after scanning: {last_ambiguous}")


def get_date_type(datetime_format: str) -> DataType | None:
    """Identifies if a datetime format string corresponds to a Date or Datetime data
    type.

    Args:
        datetime_format (str): datetime format string

    Returns:
        DataType | None: DataType of Date, Datetime, or None if not found.
    """
    config = get_config()
    return config.date_formats.get(datetime_format)
