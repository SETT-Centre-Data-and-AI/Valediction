# valediction/dictionary/generation.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
from pandas import DataFrame

from valediction.data_types.data_types import DataType
from valediction.data_types.type_inference import (
    COLUMN_STEPS,
    ColumnState,
    TypeInferer,
)
from valediction.datasets.datasets_helpers import DatasetItemLike
from valediction.dictionary.model import Column, Dictionary, Table
from valediction.io.csv_readers import (
    CsvReadConfig,
    iter_csv_chunks,
    read_csv_headers,
    read_csv_sample,
)
from valediction.progress import Progress
from valediction.support import _normalise_name, calculate_runtime

IMPORTING_DATA = "Importing data"
CHUNK_STEPS = 1
COLUMN_STEPS = COLUMN_STEPS


@dataclass(slots=True)
class GeneratorConfig:
    chunk_size: int = 10_000_000
    sample_rows: int | None = None
    dayfirst: bool = True
    infer_types: bool = True
    infer_max_length: bool = True

    def set_variables(
        self,
        chunk_size: int | None = None,
        sample_rows: int | None = None,
    ) -> None:
        # Set user variables
        self.chunk_size = chunk_size
        self.sample_rows = sample_rows


class Generator:
    """
    Summary:
        Generator class for creating dictionaries from datasets.

    Arguments:
        feedback (bool): Provide user feedback on progress (default: True)
        debug (bool): Enable debug mode, providing full log of data type inference and
            reasoning (default: False)
        chunk_size (int | None): Size of chunks for reading data to optimise RAM usage,
            if reading from CSV (default: 10_000_000)
        sample_rows (int | None): Number of rows to sample for data type inference. Note:
            this overrides `chunk_size` and reads in a single chunk (default: None)

    Raises:
        DataDictionaryError: If there is an issue with the data dictionary
    """

    def __init__(
        self,
        feedback: bool = True,
        debug: bool = False,
        chunk_size: int | None = 10_000_000,
        sample_rows: int | None = None,
    ) -> None:
        # User Config
        self.config = GeneratorConfig()
        self.config.set_variables(sample_rows=sample_rows, chunk_size=chunk_size)
        self.feedback: bool = feedback
        self.debug: bool = debug
        self.csv_cfg: CsvReadConfig = CsvReadConfig()

        # Progress
        self.progress: Progress = None

        # Setup
        if sample_rows is not None:
            self.config.sample_rows = int(sample_rows)
        if chunk_size is not None:
            self.config.chunk_size = int(chunk_size)

    def __say(
        self,
        *values: object,
        sep: str | None = " ",
        end: str | None = "\n",
    ) -> None:
        if self.feedback:
            print(*values, sep=sep, end=end)

    def generate_dictionary(
        self,
        items: Iterable[DatasetItemLike],
        dictionary_name: str | None = None,
        primary_keys: dict[str, list[str]] | None = None,
    ) -> Dictionary:
        """
        Summary:
            Generate a dictionary from a Dataset.

        Arguments:
            items (Dataset): A list of DatasetItems to generate the dictionary from.
            dictionary_name (str | None): The name of the dictionary to generate.
                If None, will not be set.
            primary_keys (dict[str, list[str]] | None): A dictionary of primary keys
                to set on the generated dictionary. If None, will not be set.

        Returns:
            Dictionary: The generated dictionary.
        """
        dictionary = Dictionary(name=dictionary_name, imported=True)

        self.__say(f"Generating dictionary for {len(items)} tables")
        for item in items:
            self.__progress_init(item)
            table = Table(name=_normalise_name(item.name))
            dictionary.add_table(table)

            if item.is_path:
                self._infer_from_csv_into_table(item, table)
            else:
                self._infer_from_dataframe_into_table(item.data, table)

            item._dictionary_runtimes = self.__finish_generation_for_table()

        dictionary.set_primary_keys(primary_keys or {})
        self.__say("\n", end="")
        return dictionary

    # Generation Helpers
    def _infer_from_csv_into_table(self, item: DatasetItemLike, table: Table) -> None:
        self.__begin_step(step=IMPORTING_DATA)
        csv_path = item.data
        inferer = TypeInferer(
            debug=self.debug,
            dayfirst=self.config.dayfirst,
            progress=self.progress,
        )

        # Read single sample
        if self.config.sample_rows is not None:
            self.__begin_step(step=IMPORTING_DATA)
            df = read_csv_sample(
                csv_path,
                nrows=self.config.sample_rows,
                cfg=self.csv_cfg,
            ).df
            self.__complete_step()

            inferer.update_with_chunk(df)
            self._create_or_update_columns(table, inferer)
            return

        # Read in chunks
        first_chunk = True
        columns_by_name: dict[str, Column] = {}
        column_count = item.column_count
        iterator = iter_csv_chunks(
            path=Path(csv_path), chunk_size=self.config.chunk_size, cfg=self.csv_cfg
        )

        while True:
            # Import chunk
            try:
                chunk = next(iterator)
            except StopIteration:
                break

            est_chunk_count = chunk.estimate_chunk_count()
            self.__progress_retarget_total(
                est_chunk_count=est_chunk_count, column_count=column_count
            )
            self.__complete_step()

            inferer.update_with_chunk(chunk.df)

            self.__begin_step(step="Saving chunk data types")
            if first_chunk:
                ordered = list(inferer.states.keys())
                for idx, col_name in enumerate(ordered, start=1):
                    col_state = inferer.states[col_name]
                    data_type, length = col_state.final_data_type_and_length()
                    col = Column(
                        name=_normalise_name(col_name),
                        order=idx,
                        data_type=data_type,
                        length=length if data_type == DataType.TEXT else None,
                        vocabulary=None,
                        primary_key=None,
                        foreign_key=None,
                        description=None,
                        enumerations=None,
                    )

                    self._set_datetime_format(column_state=col_state, column=col)
                    table.add_column(col)
                    columns_by_name[col_name] = col
                first_chunk = False

            else:
                self._apply_state_to_existing_columns(table, inferer, columns_by_name)

        if first_chunk:
            empty = read_csv_headers(
                csv_path,
                cfg=self.csv_cfg,
            )
            inferer.update_with_chunk(empty)
            self._create_or_update_columns(table, inferer)

    def _infer_from_dataframe_into_table(self, df: pd.DataFrame, table: Table) -> None:
        self.__begin_step(step=IMPORTING_DATA)
        inferer = TypeInferer(
            debug=self.debug,
            dayfirst=self.config.dayfirst,
            progress=self.progress,
        )
        self.__complete_step()

        inferer.update_with_chunk(df)
        self._create_or_update_columns(table, inferer)

    # Emit/Update Helpers
    def _create_or_update_columns(self, table: Table, inferer: TypeInferer) -> None:
        if len(table):
            for existing in table:
                table.remove_column(existing.name)

        ordered = list(inferer.states.keys())
        for idx, col_name in enumerate(ordered, start=1):
            col_state = inferer.states[col_name]
            data_type, length = col_state.final_data_type_and_length()
            col = Column(
                name=_normalise_name(col_name),
                order=idx,
                data_type=data_type,
                length=length if data_type == DataType.TEXT else None,
                vocabulary=None,
                primary_key=None,
                foreign_key=None,
                description=None,
                enumerations=None,
            )
            self._set_datetime_format(column_state=col_state, column=col)

            table.add_column(col)

    def _set_datetime_format(self, column_state: ColumnState, column: Column) -> None:
        if column.data_type in (DataType.DATE, DataType.DATETIME):
            datetime_format = getattr(column_state, "cached_datetime_format", None)
            if datetime_format and hasattr(column, "datetime_format"):
                column.datetime_format = datetime_format

        else:
            if hasattr(column, "datetime_format"):
                column.datetime_format = None

    def _apply_state_to_existing_columns(
        self,
        table: Table,
        inferer: TypeInferer,
        columns_by_name: dict[str, Column],
    ) -> None:
        for col_name, col_state in inferer.states.items():
            if col_name not in columns_by_name:
                next_order = max((c.order or 0 for c in table), default=0) + 1
                data_type, length = col_state.final_data_type_and_length()
                new_col = Column(
                    name=_normalise_name(col_name),
                    order=next_order,
                    data_type=data_type,
                    length=length if data_type == DataType.TEXT else None,
                    vocabulary=None,
                    primary_key=None,
                    foreign_key=None,
                    description=None,
                    enumerations=None,
                )
                self._set_datetime_format(column_state=col_state, column=new_col)
                table.add_column(new_col)
                columns_by_name[col_name] = new_col
                continue

            col = columns_by_name[col_name]
            data_type, length = col_state.final_data_type_and_length()

            if col.data_type != data_type:
                col.data_type = data_type

            if data_type == DataType.TEXT:
                if length is not None and (col.length or 0) < length:
                    col.length = int(length)
            else:
                col.length = None

            self._set_datetime_format(column_state=col_state, column=col)

    # Progress
    def __progress_init(self, item: DatasetItemLike) -> None:
        # Switch to debug mode
        if self.debug:
            self.progress = Progress(enabled=False)
            return

        # Switch to silent mode
        if not self.feedback:
            self.progress = Progress(enabled=False)
            return

        # Progress bars on
        total_steps = (
            (CHUNK_STEPS + (COLUMN_STEPS * item.column_count))
            if (isinstance(item.data, DataFrame) or self.config.sample_rows)
            else None
        )
        pad = " " * item._padding if item._padding else ""

        self.progress = Progress(
            desc=f"Generating {item.name}: {pad}",
            starting_step=IMPORTING_DATA,
            est_total=total_steps,
            smoothing_steps=(COLUMN_STEPS * item.column_count),
        )

    def __progress_retarget_total(
        self, est_chunk_count: int, column_count: int
    ) -> None:
        new_total = (CHUNK_STEPS * est_chunk_count) + (
            COLUMN_STEPS * est_chunk_count * column_count
        )
        self.progress.retarget_total(new_total=new_total)

    def __begin_step(self, step: str | None = None) -> None:
        self.progress.begin_step(step=step)

    def __complete_step(self) -> None:
        self.progress.complete_step()

    def __finish_generation_for_table(self) -> dict[str, timedelta]:
        step = (
            f"Completed ({calculate_runtime(start=self.progress.full_start).message})"
        )
        save_as = "Total"
        self.progress.finish(postfix=step, save_as=save_as, good=True)
        self.progress.close()
        return self.progress.runtimes
