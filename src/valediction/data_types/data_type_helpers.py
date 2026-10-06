import re
from datetime import date, datetime, timedelta
from typing import Literal

from numpy import flatnonzero
from pandas import NA, NaT, Series, Timestamp, isna, to_datetime
from pandas.api.types import is_datetime64_any_dtype

from valediction.data_types.data_types import DataType
from valediction.integrity import get_config


def _date_order(fmt: str) -> str | None:
    directives = re.findall(r"%[Yymd]", fmt)
    if "%m" not in directives or "%d" not in directives:
        return None
    if directives[:3] == ["%Y", "%m", "%d"] or directives[:3] == ["%y", "%m", "%d"]:
        return None
    return "day" if directives.index("%d") < directives.index("%m") else "month"


def _parse_offset_datetimes(column: Series, fmt: str) -> Series:
    if fmt not in {"%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"}:
        return column.map(
            lambda value: NaT
            if isna(value)
            else to_datetime(value, format=fmt, errors="coerce", utc=False),
        )

    groups: dict[timedelta | None, list[int]] = {}
    for position, value in enumerate(column):
        try:
            offset = datetime.fromisoformat(value).utcoffset()
        except (TypeError, ValueError):
            offset = None
        groups.setdefault(offset, []).append(position)

    if len(groups) <= 1:
        return to_datetime(column, format=fmt, errors="coerce", utc=False)

    result = [NaT] * len(column)
    for positions in groups.values():
        parsed = to_datetime(
            column.iloc[positions], format=fmt, errors="coerce", utc=False
        )
        for position, value in zip(positions, parsed, strict=False):
            result[position] = value
    return Series(result, index=column.index, name=column.name)


def _has_time(parsed: Series) -> Series:
    if is_datetime64_any_dtype(parsed):
        return parsed.notna() & (
            (parsed.dt.hour != 0)
            | (parsed.dt.minute != 0)
            | (parsed.dt.second != 0)
            | (parsed.dt.microsecond != 0)
            | (parsed.dt.nanosecond != 0)
        )
    return parsed.map(
        lambda value: bool(
            value.hour
            or value.minute
            or value.second
            or value.microsecond
            or value.nanosecond
        )
        if isinstance(value, Timestamp) and value is not NaT
        else False
    )


class _DatetimeResolver:
    """Retain compatible calendar conventions and positional conflict evidence."""

    def __init__(self, dtype: DataType | None = None, fmt: str | None = None):
        configured = get_config().date_formats
        formats = [
            candidate
            for candidate, kind in configured.items()
            if dtype is None
            or kind == dtype
            or (dtype == DataType.TIMESTAMP and kind == DataType.DATE)
        ]
        if fmt:
            formats.insert(0, fmt)
            if dtype == DataType.TIMESTAMP:
                formats.extend(
                    [
                        "%Y-%m-%d %H:%M:%S",
                        "%Y-%m-%d %H:%M:%S.%f",
                        "%Y-%m-%dT%H:%M:%S",
                        "%Y-%m-%dT%H:%M:%S.%f",
                        "%Y-%m-%dT%H:%M:%S%z",
                        "%Y-%m-%dT%H:%M:%S.%f%z",
                    ]
                )
        self.formats = list(dict.fromkeys(formats))
        self.dtype = dtype
        self.fmt = fmt
        self.orders = list(
            dict.fromkeys(
                _date_order(candidate)
                for candidate in self.formats
                if _date_order(candidate)
            )
        )
        self.fixed_order = _date_order(fmt) if fmt else None
        self.remaining = {self.fixed_order} if self.fixed_order else set(self.orders)
        self.evidence: dict[str, list[tuple[int, int]]] = {
            order: [] for order in self.orders
        }
        self.observed: set[str] = set()
        self.regional: set[str] = set()
        self.has_values = False
        self.has_time = False
        self.invalid = False
        self.conflict = False

    @staticmethod
    def _record_ranges(ranges: list[tuple[int, int]], mask: Series, start: int) -> None:
        for position in flatnonzero(mask.to_numpy(dtype=bool)):
            row = start + int(position)
            if ranges and ranges[-1][1] + 1 == row:
                ranges[-1] = (ranges[-1][0], row)
            else:
                ranges.append((row, row))

    def _candidates(self, column: Series) -> dict[str, Series]:
        parsed = {}
        for candidate in self.formats:
            if is_datetime64_any_dtype(column):
                values = column.copy()
            elif (
                column.notna().any()
                and column.dropna()
                .map(lambda value: isinstance(value, (date, datetime)))
                .all()
            ):
                values = column.map(Timestamp, na_action="ignore")
            elif "%z" in candidate:
                values = _parse_offset_datetimes(column, candidate)
            else:
                values = to_datetime(
                    column, format=candidate, errors="coerce", utc=False
                )
            if self.dtype == DataType.DATE:
                values = values.mask(_has_time(values))
            parsed[candidate] = values
        return parsed

    def _matches(
        self, parsed: dict[str, Series], neutral: Series
    ) -> tuple[Series, dict[str, Series]]:
        matches = {order: neutral.copy() for order in self.orders}
        primary = parsed.get(self.fmt)
        primary_ok = primary.notna() if primary is not None else neutral.copy()
        for candidate, values in parsed.items():
            ok = values.notna()
            if ok.any():
                self.observed.add(candidate)
                self.has_time |= bool(_has_time(values).any())
            order = _date_order(candidate)
            if order:
                matches[order] |= ok
            else:
                neutral |= ok
        if self.fixed_order:
            neutral &= ~primary_ok
            for order in matches:
                if order != self.fixed_order:
                    matches[order] &= ~primary_ok
        return neutral, matches

    def update(self, column: Series, start: int = 0) -> Series:
        parsed = self._candidates(column)
        neutral, matches = self._matches(parsed, Series(False, index=column.index))
        valid = neutral.copy()
        for ok in matches.values():
            valid |= ok
        invalid = column.notna() & ~valid
        self.invalid |= bool(invalid.any())
        self.has_values |= bool(column.notna().any())
        for candidate, values in parsed.items():
            if _date_order(candidate) and (values.notna() & ~neutral).any():
                self.regional.add(candidate)
        for order, ok in matches.items():
            exclusive = ok & ~neutral
            for other_order, other_ok in matches.items():
                if order != other_order:
                    exclusive &= ~other_ok
            self._record_ranges(self.evidence[order], exclusive, start)
            if (valid & ~neutral & ~ok).any():
                self.remaining.discard(order)
        if (valid & ~neutral).any() and not self.remaining:
            self.conflict = True
        return invalid

    def conflict_ranges(self) -> list[tuple[int, int]]:
        if not self.conflict:
            return []
        return sorted(
            span
            for order, ranges in self.evidence.items()
            if not self.fixed_order or order != self.fixed_order
            for span in ranges
        )

    def selected_format(self) -> str | None:
        if self.conflict or self.invalid or not self.has_values:
            return None
        if self.fmt and (self.fixed_order or not self.regional):
            return self.fmt
        order = self._selected_order()
        allowed = [
            candidate
            for candidate in self.formats
            if _date_order(candidate) is None or _date_order(candidate) == order
        ]
        observed = [candidate for candidate in allowed if candidate in self.observed]
        regional = [candidate for candidate in observed if candidate in self.regional]
        choices = regional or observed
        if self.has_time:
            choices = [
                candidate for candidate in choices if re.search(r"%[HI]", candidate)
            ] or choices
        return next(iter(choices), None)

    def _selected_order(self) -> str | None:
        if self.fixed_order:
            return self.fixed_order
        eligible = self.remaining if not self.conflict else set(self.orders)
        return next(
            (
                _date_order(candidate)
                for candidate in self.formats
                if candidate in self.regional and _date_order(candidate) in eligible
            ),
            None,
        )

    def parse(self, column: Series, *, errors: Literal["coerce", "raise"]) -> Series:
        invalid = self.update(column)
        if errors == "raise" and (self.conflict or invalid.any()):
            raise ValueError(
                "Inconsistent date interpretations"
                if self.conflict
                else "Values do not match configured datetime formats"
            )
        order = self._selected_order()
        parsed = Series(
            NaT, index=column.index, name=column.name, dtype="datetime64[ns]"
        )
        for candidate, values in self._candidates(column).items():
            if _date_order(candidate) is not None and _date_order(candidate) != order:
                continue
            unresolved = column.notna() & parsed.isna()
            if not values.notna().any():
                continue
            if not parsed.notna().any():
                parsed = values.copy()
            else:
                if parsed.dtype != values.dtype:
                    parsed = parsed.astype(object)
                positions = flatnonzero(unresolved.to_numpy(dtype=bool))
                parsed.iloc[positions] = values.iloc[positions]
        if self.conflict:
            for start, end in self.conflict_ranges():
                parsed.iloc[start : end + 1] = NaT
        return parsed


def infer_datetime_format(
    series: Series,
    slice_sample_size: int = 100,
) -> str | None:
    """Resolve a consistent calendar interpretation, using configured order for ties.

    Return a representative format, or None for invalid/conflicting values. Empty input
    raises ValueError.
    """
    values = series.str.strip().replace("", NA).dropna()
    if values.empty:
        raise ValueError("Series has no non-null values to test.")
    resolver = _DatetimeResolver()
    for start in range(0, len(values), slice_sample_size):
        resolver.update(values.iloc[start : start + slice_sample_size], start)
    return resolver.selected_format()


def get_date_type(datetime_format: str) -> DataType | None:
    """Identifies if a datetime format string corresponds to a Date or Timestamp data
    type.

    Args:
        datetime_format (str): datetime format string

    Returns:
        DataType | None: DataType of Date, Timestamp, or None if not found.
    """
    config = get_config()
    return config.date_formats.get(datetime_format)
