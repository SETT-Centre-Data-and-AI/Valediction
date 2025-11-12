from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Iterator, Optional

from pandas import DataFrame, concat

from valediction.datasets.datasets_helpers import DatasetItemLike
from valediction.io.csv_readers import CsvReadConfig, read_csv_ranges
from valediction.support import _normalise_name, list_as_bullets


class IssueType(Enum):
    # Column / schema
    MISSING_COLUMN = "MissingColumn"
    EXTRA_COLUMN = "ExtraColumn"
    FULLY_NULL_COLUMN = "FullyNullColumn"

    # Keys
    PK_NULL = "PrimaryKeyNull"
    PK_COLLISION = "PrimaryKeyCollision"
    PK_WHITESPACE = "PrimaryKeyContainsWhitespace"

    # Types / content
    TYPE_MISMATCH = "TypeMismatch"
    TEXT_TOO_LONG = "TextTooLong"
    FORBIDDEN_CHARACTER = "ForbiddenCharacter"


# Settings
APPLIES_WHOLE_COLUMN = {
    IssueType.MISSING_COLUMN,
    IssueType.EXTRA_COLUMN,
    IssueType.FULLY_NULL_COLUMN,
}

PRIMARY_KEY_ISSUES = {
    IssueType.PK_NULL,
    IssueType.PK_COLLISION,
    IssueType.PK_WHITESPACE,
}


@dataclass
class Range:
    start: int
    end: int

    def __init__(self, start: int, end: int):
        self.start: int = int(start)
        self.end: int = int(end)


@dataclass
class Issue:
    """
    Summary:
    Dataclass representing an issue in the dataset.

    Attributes:
        type (IssueType): type of issue
        table (str): name of the table where the issue was detected
        column (str | None): name of the column where the issue was detected, or None if not applicable
        ranges (list[Range]): list of contiguous ranges of rows where the issue was detected
        parent (DatasetItemLike | None): parent dataset item, or None if not applicable
    """

    type: IssueType
    table: str
    column: str | None
    ranges: list[Range] = field(default_factory=list)
    parent: DatasetItemLike | None = None

    # Magic
    def __repr__(self) -> str:
        column_part = f", column={self.column!r}" if self.column is not None else ""
        sum_ranges = sum(r.end - r.start + 1 for r in self.ranges)
        sum_range_part = f", total={sum_ranges}" if sum_ranges else ""
        return f"Issue(type={self.type.value!r}, table={self.table!r}{column_part}{sum_range_part})"

    # Methods
    def add_ranges(self, new_ranges: Iterable[Range]) -> None:
        """
        Summary:
            Merge new contiguous/overlapping ranges into self.ranges (kept sorted).

        Arguments:
            new_ranges (Iterable[Range]): new contiguous/overlapping ranges to be merged into self.ranges

        Raises:
            ValueError: if new_ranges is empty
        """
        all_ranges = self.ranges + list(new_ranges)
        if not all_ranges:
            self.ranges = []
            return
        all_ranges.sort(key=lambda r: (r.start, r.end))
        merged: list[Range] = []
        cur = all_ranges[0]
        for r in all_ranges[1:]:
            if r.start <= cur.end + 1:  # contiguous/overlap
                cur.end = max(cur.end, r.end)
            else:
                merged.append(cur)
                cur = r
        merged.append(cur)
        self.ranges = merged

    def inspect(
        self,
        additional_columns: bool | str | list[str] | None = None,
        chunk_size: int = 1_000_000,
        print_header: bool = True,
    ) -> DataFrame | str:
        """
        Summary:
            Inspect an issue in the dataset by returning a DataFrame containing the relevant values.

        Arguments:
            additional_columns (bool | str | list[str] | None): whether to include additional columns in the DataFrame
                - if True, include all columns
                - if str or list[str], include only the specified columns
                - if None, do not include any additional columns
            chunk_size (int): the number of rows to include in the DataFrame at a time
            print_header (bool): whether to print the issue details as a header

        Returns:
            DataFrame: a DataFrame containing the relevant rows of the dataset

        Raises:
            ValueError: if the issue has no parent DatasetItem
        """
        # Guard
        if not self.parent:
            raise ValueError("Issue has no parent DatasetItem")
        header = self.__repr__() if print_header else ""
        # Not applicable
        if self.type in APPLIES_WHOLE_COLUMN:
            print(f"{header}: applies to whole column")
            return None

        # Column Inclusion
        if print_header:
            print(f"{header}:")
        if additional_columns is True:
            columns = None
        else:
            additional_columns = (
                [additional_columns]
                if isinstance(additional_columns, str)
                else additional_columns
            )
            base = (
                set(self.parent.primary_keys)
                if self.type in PRIMARY_KEY_ISSUES
                else {self.column}
            )
            base |= set(additional_columns or [])
            base.discard(None)
            columns = list(base) if base else None

        if not self.ranges:
            return DataFrame(columns=columns) if columns else DataFrame()

        spans: list[tuple[int, int]] = [(r.start, r.end) for r in self.ranges]

        # DataFrame source: slice directly
        if self.parent.is_dataframe:
            df: DataFrame = self.parent.data
            n = len(df)
            if n == 0:
                return DataFrame(columns=columns) if columns else DataFrame()

            # Clamp spans to df length; build parts
            parts: list[DataFrame] = []
            for s, e in spans:
                if s > e or s >= n or e < 0:
                    continue
                lo = max(0, s)
                hi = min(n - 1, e)
                part: DataFrame = df.iloc[lo : hi + 1]
                parts.append(part if columns is None else part.loc[:, columns])

            if not parts:
                return DataFrame(columns=columns) if columns else DataFrame()
            return concat(parts, axis=0, ignore_index=False)

        # CSV source: delegate reading to csv_readers
        if self.parent.is_path:
            path = self.parent.data
            cfg = CsvReadConfig(usecols=columns)
            out = read_csv_ranges(path, spans, cfg=cfg, chunk_size=chunk_size)

        return out if columns is None else out.loc[:, columns]


@dataclass
class Issues:
    """List-like container holding Issues with case-insensitive get and range
    merging."""

    # Magic
    def __init__(self) -> None:
        self._items: list[Issue] = []
        self._index: dict[
            tuple[str, Optional[str], IssueType], Issue
        ] = {}  # table, column, issue_type

    def __iter__(self) -> Iterator[Issue]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __getitem__(self, idx) -> Issue | list[Issue]:
        return self._items[idx]

    def __repr__(self) -> str:
        if not self._items:
            return "Issues([])"
        issues = list_as_bullets(elements=[repr(item) for item in self._items])
        return f"Issues({issues}\n)"

    # Methods
    def add(
        self,
        issue_type: IssueType,
        table: str,
        column: str | None = None,
        ranges: Iterable[Range] | None = None,
        parent: DatasetItemLike | None = None,
    ) -> Issue:
        key = (
            _normalise_name(table),
            _normalise_name(column) if column is not None else None,
            issue_type,
        )
        issue = self._index.get(key)
        if issue is None:
            issue = Issue(type=issue_type, table=table, column=column, parent=parent)
            self._items.append(issue)
            self._index[key] = issue
        if ranges:
            issue.add_ranges(ranges)
        return issue

    def get(
        self,
        table: str,
        column: str | None = None,
        issue_type: IssueType | None = None,
    ) -> list[Issue]:
        """Case-insensitive filter; any arg can be None to act as a wildcard."""
        table = _normalise_name(table)
        column = _normalise_name(column) if column is not None else None
        output: list[Issue] = []
        if issue_type is not None:
            # direct index lookup where possible
            key = (table, column, issue_type)
            hit = self._index.get(key)
            if hit:
                output.append(hit)
            return output

        # otherwise scan (still cheap; we maintain a compact list)
        for item in self._items:
            if _normalise_name(item.table) != table:
                continue
            if column is not None and (_normalise_name(item.column) or "") != column:
                continue
            output.append(item)
        return output

    def extend(self, issues: Issues) -> None:
        for issue in issues:
            self.add(issue.type, issue.table, issue.column, issue.ranges, issue.parent)
