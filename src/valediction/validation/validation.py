from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
from pandas import DataFrame, Series

from valediction.data_types.data_type_helpers import (
    infer_datetime_format,
)
from valediction.data_types.data_types import DataType
from valediction.datasets.datasets_helpers import DataLike, DatasetItemLike
from valediction.dictionary.model import Table
from valediction.exceptions import DataDictionaryImportError, DataIntegrityError
from valediction.io.csv_readers import (
    CsvReadConfig,
    FrameChunk,
    iter_csv_chunks,
)
from valediction.progress import Progress
from valediction.support import _get_runtime_string, calculate_runtime
from valediction.validation.helpers import (
    _column_has_values,
    _set_nulls,
    create_pk_hashes,
    invalid_mask_date,
    invalid_mask_datetime,
    invalid_mask_float,
    invalid_mask_integer,
    invalid_mask_text_forbidden_characters,
    invalid_mask_text_too_long,
    mask_to_ranges,
    pk_contains_whitespace_mask,
)
from valediction.validation.issues import Issues, IssueType, Range

IMPORTING_DATA = "Importing data"
SINGLE_STEPS: int = 3  # tweak if adding/amending step tracking
CHUNK_STEPS: int = 12  # tweak if adding/amending step tracking


class Validator:
    """
    Summary:
    Validates a dataset against a dictionary.

    Arguments:
    dataset_item (DatasetItemLike): dataset item to validate
    table_dictionary (Table): table dictionary to validate against
    feedback (bool): whether to provide feedback on validation (default: True)
    chunk_size (int): size of chunks to validate (default: 10_000_000)

    Raises:
    DataDictionaryImportError: if the provided dictionary is invalid
    DataIntegrityError: if the validated dataset contains invalid values
    """

    def __init__(
        self,
        dataset_item: DatasetItemLike,
        table_dictionary: Table,
        feedback: bool = True,
        chunk_size: int | None = 10_000_000,
        _padding: int = 0,
    ):
        # User Variables
        self.dataset_item = dataset_item
        self.data: DataLike = dataset_item.data
        self.table_dictionary: Table = table_dictionary
        self.chunk_size: int = chunk_size
        self.feedback: bool = feedback
        self._padding: int = _padding

        # Config
        self.table_name: str = None
        self.issues: Issues = None
        self.csv_cfg: CsvReadConfig = None

        # Validation Tracking
        self.tracker_seen_non_nulls: dict[str, bool] = {}
        self.tracker_pk_hashes: dict[int, int] = {}
        self.tracker_pk_reported_first: set[int] = set()
        self._dt_format_cache: dict[str, str | None] = {}
        self._dt_needs_infer: set[str] = set()

        #  Helpers
        self._column_names: set = set(self.table_dictionary.get_column_names())

        # Progress Tracking
        self.progress: Progress | None = None
        self.est_chunk_count: int = None
        self._runtimes: dict[str, timedelta] = None

        # Setup
        self.__check_dictionary()
        self.__init_issues()
        self.__init_csv_cfg()
        self.__reset_pk_trackers()
        self.__import_datetime_format_cache()

    # Properties
    @property
    def data_is_path(self) -> bool:
        return isinstance(self.data, (Path, str))

    @property
    def data_is_dataframe(self) -> bool:
        return isinstance(self.data, DataFrame)

    @property
    def runtimes(self) -> dict[str, str]:
        return {
            step: _get_runtime_string(time_delta)
            for step, time_delta in self._runtimes.items()
        }

    # Initialisation
    def __check_dictionary(self):
        if self.table_dictionary is None or not isinstance(
            self.table_dictionary, Table
        ):
            raise DataDictionaryImportError(
                "Data Dictionary not yet imported or generated. "
                + "Validation must first have a Data Dictionary. "
                + "Please first run DataSet.import_dictionary(), including `primary_keys`."
            )

        self.table_dictionary.check()
        self.table_name = self.table_dictionary.name

    def __init_issues(self):
        self.issues = Issues()

    def __init_csv_cfg(self):
        self.csv_cfg = CsvReadConfig()

    def __reset_pk_trackers(self):
        self.tracker_pk_hashes: dict[int, int] = {}
        self.tracker_pk_reported_first: set[int] = set()

    def __import_datetime_format_cache(self) -> None:
        self._dt_format_cache.clear()
        self._dt_needs_infer.clear()

        for column in self.table_dictionary:
            name = column.name
            datetime_format = column.datetime_format
            data_type = column.data_type

            if data_type in (DataType.DATE, DataType.DATETIME):
                self._dt_format_cache[name] = datetime_format

                if not datetime_format:
                    self._dt_needs_infer.add(name)

    # Validate
    def validate(self):
        """
        Summary:
            Validate the dataset against the data dictionary

        Raises:
            DataDictionaryImportError: if the data dictionary has not yet been imported or generated
            DataDictionaryError: if the data dictionary contains invalid data
            ValueError: if the dataset contains invalid data
        """
        self.__progress_init()
        first_chunk = True

        for chunk in self.__iterate_data_chunks(
            self.data, self.chunk_size, self.csv_cfg
        ):
            df = chunk.df
            start = chunk.start

            # First Chunk Only
            if first_chunk:
                _ = self._check_for_missing_columns(df)
                _ = self._check_for_extra_columns(df)
                first_chunk = False

            # Remove Nulls
            df = self._set_nulls(df)

            # Structural Checks
            self._check_for_column_nulls(df)
            self._check_primary_key_whitespace(df, start_row=start)
            self._check_primary_key_integrity(df, start_row=start)

            # Data Type Checks
            self._infer_datetime_formats(df)
            self._check_column_types(df, start_row=start)
            self._check_text_lengths(df, start_row=start)
            self._check_text_forbidden_chars(df, start_row=start)

        # Final Checks
        self._check_for_fully_null_column()

        # Finish:
        self.__reset_pk_trackers()
        self.__finish_validation()
        if self.issues:
            raise DataIntegrityError(self.issues)

    # Global Helpers

    def __iterate_data_chunks(
        self,
        data: Path | str | DataFrame,
        chunk_size: int | None,
        csv_config: CsvReadConfig | None = None,
    ) -> Iterator[FrameChunk]:
        """Yield FrameChunk and keep a running estimate of total chunks for the progress
        bar."""
        csv_config = csv_config or CsvReadConfig()
        # In-memory DataFrame: single chunk
        if isinstance(data, DataFrame):
            self.__begin_step(step=IMPORTING_DATA)
            n = len(data)
            if n == 0:
                self.__complete_step()
                return

            # One chunk only
            self.__progress_retarget_total(est_chunk_count=1)

            self.__complete_step()
            yield FrameChunk(
                df=data,
                start=0,
                end=n - 1,
                total_size=None,
                file_pos=None,
                bytes_read=None,
                chunk_index=1,
                total_bytes_read=None,
                total_chunks_seen=None,
            )
            return

        # Path/str: chunking

        iterator = iter_csv_chunks(
            path=Path(data), chunk_size=chunk_size, cfg=csv_config
        )
        while True:
            self.__begin_step(step=IMPORTING_DATA)
            try:
                chunk = next(iterator)
            except StopIteration:
                break

            est_chunk_count = chunk.estimate_chunk_count()
            self.__progress_retarget_total(est_chunk_count=est_chunk_count)

            # Bookkeeping & yield
            self.__complete_step()
            yield chunk

    # Finder Helpers
    def _find_data_type(self, column_name: str) -> DataType:
        return self.table_dictionary.get_column(column_name).data_type

    def _find_datetime_format(self, column_name: str) -> str:
        return self.table_dictionary.get_column(column_name).datetime_format

    def _find_max_length(self, column_name: str) -> int | None:
        return self.table_dictionary.get_column(column_name).length

    # Validation: Start Helpers
    def _check_for_missing_columns(self, df: DataFrame):
        self.__begin_step(step="Checking for missing columns")
        missing = self._column_names - set(df.columns)
        if missing:
            for column in missing:
                self.issues.add(
                    issue_type=IssueType.MISSING_COLUMN,
                    table=self.table_name,
                    column=column,
                    parent=self.dataset_item,
                )
        self.__complete_step()

    def _check_for_extra_columns(self, df: DataFrame):
        self.__begin_step(step="Checking for extra columns")
        extra = set(df.columns) - self._column_names
        if extra:
            for column in extra:
                self.issues.add(
                    issue_type=IssueType.EXTRA_COLUMN,
                    table=self.table_name,
                    column=column,
                    parent=self.dataset_item,
                )
        self.__complete_step()

    # Validation: Chunk Helpers
    def _set_nulls(self, df: DataFrame) -> DataFrame:
        self.__begin_step(step="Setting nulls")
        df = _set_nulls(df)
        self.__complete_step()
        return df

    def _check_for_column_nulls(self, df: DataFrame) -> None:
        self.__begin_step(step="Checking for column nulls")
        for column in df.columns:
            # Check if previously checked and found
            seen_or_found = self.tracker_seen_non_nulls.get(column, False)
            if not seen_or_found:
                self.tracker_seen_non_nulls[column] = _column_has_values(df[column])
        self.__complete_step()

    def _check_primary_key_whitespace(self, df: DataFrame, start_row: int) -> None:
        pk_cols = self.table_dictionary.get_primary_keys()
        if not pk_cols:
            return

        # Check for whitespace (text cols only)
        self.__begin_step(step="Checking for primary key whitespace")
        pk_cols_text = []
        for column in self.table_dictionary:
            if column.name in pk_cols and column.data_type in [DataType.TEXT]:
                pk_cols_text.append(column.name)

        if pk_cols_text:
            space_mask = pk_contains_whitespace_mask(df[pk_cols_text])
            if space_mask.any():
                self.issues.add(
                    issue_type=IssueType.PK_WHITESPACE,
                    table=self.table_name,
                    column=None,
                    ranges=mask_to_ranges(space_mask, start_row),
                    parent=self.dataset_item,
                )
        self.__complete_step()

    def _check_primary_key_integrity(self, df, start_row: int) -> None:
        pk_cols = self.table_dictionary.get_primary_keys()
        if not pk_cols:
            return

        # Create primary key hashes
        self.__begin_step(step="Creating primary key hashes")
        pk_hashes = create_pk_hashes(df[pk_cols])
        self.__complete_step()

        # Primary Key Nulls
        self.__begin_step(step="Checking for primary key nulls")
        null = pk_hashes.isna()
        non_null = ~null
        pk_hashes_non_null = pk_hashes[non_null]

        if null.any():
            self.issues.add(
                IssueType.PK_NULL,
                table=self.table_name,
                column=None,
                ranges=mask_to_ranges(null, start_row),
                parent=self.dataset_item,
            )
        self.__complete_step()

        # 2) In-chunk collisions
        self.__begin_step(step="Checking for primary key collision")

        codes, uniques = pk_hashes_non_null.factorize(sort=False)
        counts = np.bincount(codes, minlength=len(uniques))
        in_chunk_local = counts[codes] > 1
        in_chunk_collision = non_null.copy()
        in_chunk_collision.loc[non_null] = in_chunk_local

        # 3) Cross-chunk collisions
        seen_before = set(self.tracker_pk_hashes)
        unique_in_seen = np.fromiter(
            (unique in seen_before for unique in uniques),
            dtype=bool,
            count=len(uniques),
        )
        seen_before_local = unique_in_seen[codes]
        cross_chunk_local = seen_before_local & ~in_chunk_local
        cross_chunk_collision = non_null.copy()
        cross_chunk_collision.loc[non_null] = cross_chunk_local

        # 4) Valid in-chunk PKs:
        first_in_chunk_local = ~pk_hashes_non_null.duplicated(keep="first")
        first_appearance_local = first_in_chunk_local & ~seen_before_local

        # 7) Emit in-chunk collisions Issues
        if in_chunk_collision.any():
            self.issues.add(
                IssueType.PK_COLLISION,
                table=self.table_name,
                column=None,
                ranges=mask_to_ranges(in_chunk_collision, start_row),
                parent=self.dataset_item,
            )

        # 7) Emit cross-chunk collisions Issues
        if cross_chunk_collision.any():
            self.issues.add(
                IssueType.PK_COLLISION,
                table=self.table_name,
                column=None,
                ranges=mask_to_ranges(cross_chunk_collision, start_row),
                parent=self.dataset_item,
            )

            # Add the original PK row as a collision
            for h in pk_hashes_non_null[cross_chunk_local].unique():
                if h not in self.tracker_pk_reported_first:
                    first_row = self.tracker_pk_hashes[int(h)]
                    self.issues.add(
                        IssueType.PK_COLLISION,
                        table=self.table_name,
                        column=None,
                        ranges=[Range(first_row, first_row)],
                        parent=self.dataset_item,
                    )
                    self.tracker_pk_reported_first.add(int(h))
        self.__complete_step()

        # 7) Record valid PKs
        self.__begin_step(step="Caching primary keys")
        if first_appearance_local.any():
            pos = np.flatnonzero(first_appearance_local.to_numpy())
            vals = pk_hashes_non_null.to_numpy()[pos]
            start_rows = start_row + pos
            for h, r in zip(vals, start_rows, strict=False):
                self.tracker_pk_hashes.setdefault(int(h), int(r))
        self.__complete_step()

    def _infer_datetime_formats(self, df: DataFrame) -> None:
        self.__begin_step(step="Inferring datetime formats")
        if not self._dt_needs_infer:
            self.__complete_step()
            return

        columns = [col for col in self._dt_needs_infer if col in df.columns]
        if not columns:
            self.__complete_step()
            return

        for column in columns:
            series = df[column].astype("string", copy=False).str.strip()
            unique = series.dropna().unique()
            if len(unique) == 0:
                continue

            try:
                fmt_or_false = infer_datetime_format(Series(unique, dtype="string"))
            except ValueError:
                # ambiguous - try again in later chunk
                continue

            if fmt_or_false and fmt_or_false is not False:
                col_dtype = self._find_data_type(column)
                from valediction.validation.helpers import _allowed_formats_for

                allowed = _allowed_formats_for(col_dtype)
                if fmt_or_false in allowed:
                    self._dt_format_cache[column] = fmt_or_false
                    self._dt_needs_infer.discard(column)

                    # Persist in the dictionary
                    try:
                        self.table_dictionary.get_column(
                            column
                        ).datetime_format = fmt_or_false
                    except Exception:
                        pass
        self.__complete_step()

    def _check_column_types(self, df: DataFrame, start_row: int) -> None:
        self.__begin_step(step="Checking column types")
        present = [col for col in df.columns if col in self._column_names]
        for col in present:
            dtype = self._find_data_type(col)
            if dtype == DataType.TEXT:
                continue

            series = df[col]
            if dtype == DataType.INTEGER:
                invalid = invalid_mask_integer(series)
            elif dtype == DataType.FLOAT:
                invalid = invalid_mask_float(series)
            elif dtype == DataType.DATE:
                fmt = self._dt_format_cache.get(col) or self._find_datetime_format(col)
                invalid = invalid_mask_date(series, fmt)
            elif dtype == DataType.DATETIME:
                fmt = self._dt_format_cache.get(col) or self._find_datetime_format(col)
                invalid = invalid_mask_datetime(series, fmt)
            else:
                continue

            if invalid.any():
                self.issues.add(
                    IssueType.TYPE_MISMATCH,
                    table=self.table_name,
                    column=col,
                    ranges=mask_to_ranges(invalid, start_row),
                    parent=self.dataset_item,
                )
        self.__complete_step()

    def _check_text_lengths(self, df: DataFrame, start_row: int) -> None:
        self.__begin_step(step="Checking text lengths")
        present = [col for col in df.columns if col in self._column_names]
        for col in present:
            if self._find_data_type(col) != DataType.TEXT:
                continue
            max_len = self._find_max_length(col)
            invalid = invalid_mask_text_too_long(df[col], max_len)
            if invalid.any():
                self.issues.add(
                    IssueType.TEXT_TOO_LONG,
                    table=self.table_name,
                    column=col,
                    ranges=mask_to_ranges(invalid, start_row),
                    parent=self.dataset_item,
                )
        self.__complete_step()

    def _check_text_forbidden_chars(self, df: DataFrame, start_row: int) -> None:
        self.__begin_step(step="Checking for forbidden characters")
        present = [col for col in df.columns if col in self._column_names]
        for col in present:
            if self._find_data_type(col) != DataType.TEXT:
                continue
            mask = invalid_mask_text_forbidden_characters(df[col])
            if mask.any():
                self.issues.add(
                    IssueType.FORBIDDEN_CHARACTER,
                    table=self.table_name,
                    column=col,
                    ranges=mask_to_ranges(mask, start_row),
                    parent=self.dataset_item,
                )
        self.__complete_step()

    # Validation: Final Helpers
    def _check_for_fully_null_column(self):
        self.__begin_step(step="Checking for fully null columns")
        for column, seen in self.tracker_seen_non_nulls.items():
            if not seen:
                self.issues.add(
                    issue_type=IssueType.FULLY_NULL_COLUMN,
                    table=self.table_name,
                    column=column,
                    parent=self.dataset_item,
                )
        self.__complete_step()

    # Progress Helpers
    def __progress_init(self) -> None:
        if not self.feedback:
            self.progress = Progress(enabled=False)
            return

        total_steps = (
            (SINGLE_STEPS + CHUNK_STEPS)
            if (isinstance(self.data, DataFrame) or not self.chunk_size)
            else None
        )
        self.est_chunk_count = None
        pad = " " * self._padding if self._padding else ""

        self.progress = Progress(
            desc=f"Validating {self.table_name}: {pad}",
            starting_step=IMPORTING_DATA,
            est_total=total_steps,
            smoothing_steps=CHUNK_STEPS,
        )

    def __progress_retarget_total(self, est_chunk_count: int) -> None:
        """Once est_chunk_count is known, resize the bar without losing progress."""
        if est_chunk_count != self.est_chunk_count:
            self.est_chunk_count = est_chunk_count
            new_total = SINGLE_STEPS + (CHUNK_STEPS * self.est_chunk_count)
            self.progress.retarget_total(new_total=new_total)

    def __finish_validation(self) -> None:
        completed = "Completed with issues" if self.issues else "Completed"
        step = (
            f"{completed} ({calculate_runtime(start=self.progress.full_start).message})"
        )
        save_as = "Total"
        good = False if self.issues else True
        self.progress.finish(postfix=step, save_as=save_as, good=good)
        self._runtimes = self.progress.runtimes
        self.progress.close()

    def __begin_step(self, step: str | None = None) -> None:
        self.progress.begin_step(step=step)

    def __complete_step(self) -> None:
        self.progress.complete_step()
