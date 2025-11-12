from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from valediction.data_types.data_types import DataType
from valediction.dictionary.helpers import (
    _get_required_header,
    _is_missing,
    _norm_header_map,
    _normalise_name,
    _parse_int,
    _parse_truthy,
    _row_is_blank,
)
from valediction.dictionary.integrity import REQUIRED_SHEETS
from valediction.dictionary.model import Column, Dictionary, Table
from valediction.exceptions import DataDictionaryError, DataDictionaryImportError
from valediction.support import list_as_bullets


@dataclass
class _ColumnInputs:
    table_name: str
    column_name: str
    order_int: int
    data_type: DataType
    length_int: int | None
    vocabulary: str | None
    primary_key_int: int | None
    foreign_key_target: str | None
    description: str | None
    has_enumerations: bool
    row_context: str


class ExcelDataDictionary:
    def __init__(self, filepath: str | Path):
        self.path = Path(filepath)
        self.excel_file: pd.ExcelFile | None = None
        self.sheet_map: dict[str, str] = {}

        # parsed artefacts
        self.details: dict[str, Any] = {}
        self.table_metadata: dict[str, str | None] = {}
        self.enumerations: dict[tuple[str, str], dict[Any, Any]] = {}
        self.enum_flags: set[tuple[str, str]] = set()
        self.table_columns: dict[str, list[Column]] = {}
        self.tables: list[Table] = []

    # Public API
    def to_dictionary(self) -> Dictionary:
        self._open_workbook()
        try:
            self._load_sheet_map()
            self._parse_details()
            self._parse_tables()
            self._parse_enumerations()
            self._parse_columns()
            self._validate_enum_flags()
            self._build_tables()
            self._validate_foreign_keys()
            self.excel_file.close()

            return Dictionary(
                name=(self.details.get("name") or None),
                tables=self.tables,
                organisations=(self.details.get("organisations") or None),
                version=(self.details.get("version") or None),
                version_notes=(self.details.get("version_notes") or None),
                inclusion_criteria=(self.details.get("inclusion_criteria") or None),
                exclusion_criteria=(self.details.get("exclusion_criteria") or None),
                imported=True,
            )
        except Exception as error:
            self.excel_file.close()
            raise error

    # Import & Helpers
    def _open_workbook(self) -> None:
        if not self.path.exists():
            raise DataDictionaryImportError(f"File not found: {self.path}")
        try:
            self.excel_file = pd.ExcelFile(self.path, engine="openpyxl")
        except Exception as e:
            raise DataDictionaryImportError(
                f"Unable to open Excel file {self.path}: {e}"
            ) from e

    def _load_sheet_map(self) -> None:
        assert self.excel_file is not None
        self.sheet_map = {
            sheet_name.strip().lower(): sheet_name
            for sheet_name in self.excel_file.sheet_names
            if isinstance(sheet_name, str)
        }
        missing = sorted(REQUIRED_SHEETS - set(self.sheet_map))
        if missing:
            raise DataDictionaryImportError(
                "Missing sheet(s): "
                + ", ".join(missing)
                + f". Found sheets: {', '.join(self.excel_file.sheet_names)}"
            )

    def _read_sheet(self, key: str, **kwargs: Any) -> pd.DataFrame:
        assert self.excel_file is not None
        if key not in self.sheet_map:
            raise DataDictionaryImportError(f"Sheet {key!r} not found in workbook.")
        return pd.read_excel(self.excel_file, sheet_name=self.sheet_map[key], **kwargs)

    # Parse Details
    def _parse_details(self) -> None:
        details_df = self._read_sheet(
            "details", header=None, dtype=str, keep_default_na=False
        )
        keys = {
            "name": "name",
            "organisation(s)": "organisations",
            "version": "version",
            "version notes": "version_notes",
            "inclusion criteria": "inclusion_criteria",
            "exclusion criteria": "exclusion_criteria",
        }
        self.details = {v: None for v in keys.values()}
        for _, row in details_df.iterrows():
            if len(row) < 2 or not isinstance(row.iloc[0], str):
                continue
            key_norm = row.iloc[0].strip().lower()
            if key_norm in keys:
                self.details[keys[key_norm]] = row.iloc[1] or None

    # Parse Tables
    def _parse_tables(self) -> None:
        tables_df = self._read_sheet("tables", dtype=str, keep_default_na=False)
        header_map = _norm_header_map(tables_df.columns)
        table_col_header = _get_required_header(header_map, "table")
        description_col_header = _get_required_header(header_map, "description")

        meta: dict[str, str | None] = {}
        for _, row in tables_df.iterrows():
            if _is_missing(row[table_col_header]):
                continue
            table_name = _normalise_name(str(row[table_col_header]))
            table_description = (
                None
                if _is_missing(row[description_col_header])
                else str(row[description_col_header])
            )
            if table_name in meta:
                raise DataDictionaryImportError(
                    f"Duplicate table '{table_name}' in Tables sheet."
                )
            meta[table_name] = table_description
        if not meta:
            raise DataDictionaryImportError(
                "Data Dictionary sheet 'Tables' contains no table rows."
            )
        self.table_metadata = meta

    # Parse Enumerations
    def _parse_enumerations(self) -> None:
        enums_df = self._read_sheet("enumerations", dtype=str)
        header_map = _norm_header_map(enums_df.columns)
        table_col_header = _get_required_header(header_map, "table")
        column_col_header = _get_required_header(header_map, "column")
        code_col_header = _get_required_header(header_map, "code")
        name_col_header = _get_required_header(header_map, "name")

        enum_map: dict[tuple[str, str], dict[Any, Any]] = {}
        for _, row in enums_df.iterrows():
            if (
                _is_missing(row[table_col_header])
                or _is_missing(row[column_col_header])
                or _is_missing(row[code_col_header])
            ):
                continue
            table_name = _normalise_name(str(row[table_col_header]))
            column_name = _normalise_name(str(row[column_col_header]))
            enum_map.setdefault((table_name, column_name), {})
            enum_map[(table_name, column_name)][row[code_col_header]] = row[
                name_col_header
            ]
        self.enumerations = enum_map

    # Parse Columns
    def _parse_columns(self) -> None:
        columns_df, header_map = self._prepare_columns_sheet()
        (
            table_col_header,
            column_col_header,
            order_col_header,
            data_type_col_header,
            length_col_header,
            vocabulary_col_header,
            enumeration_flag_col_header,
            primary_key_col_header,
            foreign_key_col_header,
            description_col_header,
        ) = self._columns_headers(header_map)

        self.table_columns = {t: [] for t in self.table_metadata}
        errors: list[str] = []

        for idx, row in columns_df.iterrows():
            if _row_is_blank(row, (table_col_header, column_col_header)):
                continue
            try:
                inputs = self._extract_column_inputs(
                    idx=idx,
                    row=row,
                    table_col_header=table_col_header,
                    column_col_header=column_col_header,
                    order_col_header=order_col_header,
                    data_type_col_header=data_type_col_header,
                    length_col_header=length_col_header,
                    vocabulary_col_header=vocabulary_col_header,
                    enumeration_flag_col_header=enumeration_flag_col_header,
                    primary_key_col_header=primary_key_col_header,
                    foreign_key_col_header=foreign_key_col_header,
                    description_col_header=description_col_header,
                )
            except DataDictionaryImportError as e:
                errors.append(str(e))
                continue

            try:
                column_obj = self._make_column(inputs)
            except DataDictionaryError as e:
                errors.append(f"{inputs.row_context}: {e}")
                continue

            self.table_columns[inputs.table_name].append(column_obj)
            if inputs.has_enumerations:
                self.enum_flags.add((inputs.table_name, inputs.column_name))

        if errors:
            raise DataDictionaryImportError(
                "Errors while parsing Columns sheet:\n" + list_as_bullets(errors)
            ) from None

    # Validate Enumeration Flags
    def _validate_enum_flags(self) -> None:
        missing: list[str] = []
        for key in self.enum_flags:
            if key not in self.enumerations or not self.enumerations[key]:
                table_name, column_name = key
                missing.append(
                    f"{table_name}.{column_name} marked as having enumerations but none defined in Enumerations sheet."
                )
        if missing:
            # Template issue, not model construction → ImportError
            raise DataDictionaryImportError(
                "Missing enumerations:\n" + list_as_bullets(missing)
            )

    # Build Tables
    def _build_tables(self) -> None:
        self.tables = []
        for table_name, table_description in self.table_metadata.items():
            columns_for_table = self.table_columns.get(table_name, [])
            if not columns_for_table:
                raise DataDictionaryImportError(
                    f"Table '{table_name}' has no columns defined in Columns sheet."
                )
            try:
                self.tables.append(
                    Table(
                        name=table_name,
                        description=table_description,
                        columns=columns_for_table,
                    )
                )
            except DataDictionaryError as e:
                # model-level errors (e.g., duplicate orders inside Table) bubble as DataDictionaryError
                raise DataDictionaryImportError(f"In table {table_name!r}: {e}") from e

    # Validate Foreign Keys
    def _validate_foreign_keys(self) -> None:
        name_to_table = {t.name: t for t in self.tables}
        errors: list[str] = []
        for table in self.tables:
            for column in table:
                if not column.foreign_key:
                    continue
                target = column.foreign_key.strip()
                if target.count(".") != 1:
                    errors.append(
                        f"{table.name}.{column.name} foreign key must be 'TABLE.COLUMN' (got {target!r})."
                    )
                    continue
                target_table_raw, target_column_raw = target.split(".", 1)
                target_table_name = _normalise_name(target_table_raw)
                target_column_name = _normalise_name(target_column_raw)
                referenced_table = name_to_table.get(target_table_name)
                if not referenced_table:
                    errors.append(
                        f"{table.name}.{column.name} references unknown table {target_table_name!r}."
                    )
                    continue
                try:
                    referenced_table.get_column(target_column_name)
                except KeyError:
                    errors.append(
                        f"{table.name}.{column.name} references unknown column {target_table_name}.{target_column_name}."
                    )
        if errors:
            # Template issue → ImportError
            raise DataDictionaryImportError(
                "Foreign key validation errors:\n" + list_as_bullets(errors)
            )

    # Parse Columns Helpers
    def _prepare_columns_sheet(self) -> tuple[pd.DataFrame, dict[str, str]]:
        columns_df = self._read_sheet("columns", dtype=str, keep_default_na=False)
        header_map = _norm_header_map(columns_df.columns)
        # ensure required headers exist
        for key in ("table", "column", "order", "data_type"):
            _get_required_header(header_map, key)
        return columns_df, header_map

    def _columns_headers(
        self, header_map: dict[str, str]
    ) -> tuple[
        str,
        str,
        str,
        str,
        str | None,
        str | None,
        str | None,
        str | None,
        str | None,
        str | None,
    ]:
        table_col_header = _get_required_header(header_map, "table")
        column_col_header = _get_required_header(header_map, "column")
        order_col_header = _get_required_header(header_map, "order")
        data_type_col_header = _get_required_header(header_map, "data_type")

        length_col_header = header_map.get("length")
        vocabulary_col_header = header_map.get("vocabularies")
        enumeration_flag_col_header = header_map.get("enumerations")
        primary_key_col_header = header_map.get("primary_key")
        foreign_key_col_header = header_map.get("foreign_key_target")
        description_col_header = header_map.get("description")
        return (
            table_col_header,
            column_col_header,
            order_col_header,
            data_type_col_header,
            length_col_header,
            vocabulary_col_header,
            enumeration_flag_col_header,
            primary_key_col_header,
            foreign_key_col_header,
            description_col_header,
        )

    def _extract_column_inputs(
        self,
        *,
        idx: int,
        row: pd.Series,
        table_col_header: str,
        column_col_header: str,
        order_col_header: str,
        data_type_col_header: str,
        length_col_header: str | None,
        vocabulary_col_header: str | None,
        enumeration_flag_col_header: str | None,
        primary_key_col_header: str | None,
        foreign_key_col_header: str | None,
        description_col_header: str | None,
    ) -> _ColumnInputs:
        row_context = f"(Columns row {idx + 2})"

        # Required presence
        missing_fields: list[str] = []
        if _is_missing(row[table_col_header]):
            missing_fields.append("Table")
        if _is_missing(row[column_col_header]):
            missing_fields.append("Column")
        if _is_missing(row[data_type_col_header]):
            missing_fields.append("Data Type")
        if _is_missing(row[order_col_header]):
            missing_fields.append("Order")
        if missing_fields:
            raise DataDictionaryImportError(
                f"{row_context}: missing required field(s): {', '.join(missing_fields)}."
            )

        table_name = _normalise_name(str(row[table_col_header]))
        column_name = _normalise_name(str(row[column_col_header]))
        if table_name not in self.table_metadata:
            raise DataDictionaryImportError(
                f"{row_context}: Table '{table_name}' not present in Tables sheet."
            )

        order_int = _parse_int(row[order_col_header], "Order", row_context)
        length_int = (
            _parse_int(row[length_col_header], "Length", row_context, required=False)
            if length_col_header
            else None
        )
        primary_key_int = (
            _parse_int(
                row[primary_key_col_header], "Primary Key", row_context, required=False
            )
            if primary_key_col_header
            else None
        )

        vocabulary = (
            str(row[vocabulary_col_header]).strip()
            if (vocabulary_col_header and not _is_missing(row[vocabulary_col_header]))
            else None
        )
        foreign_key_target = (
            str(row[foreign_key_col_header]).strip()
            if (foreign_key_col_header and not _is_missing(row[foreign_key_col_header]))
            else None
        )
        description = (
            str(row[description_col_header]).strip()
            if (description_col_header and not _is_missing(row[description_col_header]))
            else None
        )

        try:
            data_type = DataType.parse(str(row[data_type_col_header]))
        except Exception as e:
            raise DataDictionaryImportError(
                f"{row_context}: invalid Data Type {row[data_type_col_header]!r}: {e}"
            ) from e

        has_enumerations = (
            _parse_truthy(row[enumeration_flag_col_header])
            if (
                enumeration_flag_col_header
                and not _is_missing(row[enumeration_flag_col_header])
            )
            else False
        )

        return _ColumnInputs(
            table_name=table_name,
            column_name=column_name,
            order_int=order_int,
            data_type=data_type,
            length_int=length_int,
            vocabulary=vocabulary,
            primary_key_int=primary_key_int,
            foreign_key_target=foreign_key_target,
            description=description,
            has_enumerations=has_enumerations,
            row_context=row_context,
        )

    def _make_column(self, inputs: _ColumnInputs) -> Column:
        enums_for_column = self.enumerations.get(
            (inputs.table_name, inputs.column_name), {}
        )
        return Column(
            name=inputs.column_name,
            order=inputs.order_int,
            data_type=inputs.data_type,
            length=inputs.length_int,
            vocabulary=inputs.vocabulary,
            primary_key=inputs.primary_key_int,
            foreign_key=inputs.foreign_key_target,
            description=inputs.description,
            enumerations=enums_for_column,
        )


# Public Entry
def import_dictionary(filepath: str | Path) -> Dictionary:
    """
    Summary:
    Import an Excel data dictionary into a Python Dictionary object.

    Args:
        filepath (str | Path): Path to the Excel data dictionary file.

    Returns:
        Dictionary: A Python Dictionary object created from the Excel data dictionary.

    Raises:
        DataDictionaryImportError: If there is an error importing the data dictionary.
    """
    return ExcelDataDictionary(filepath).to_dictionary()
