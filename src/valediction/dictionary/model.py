from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from valediction.data_types.data_types import DataType
from valediction.dictionary.helpers import (
    _check_data_type,
    _check_name,
    _check_order,
    _check_primary_key,
    _normalise_name,
)
from valediction.exceptions import DataDictionaryError
from valediction.support import list_as_bullets


class Column:
    """Represents a single column in a data dictionary.

    Attributes:
        name (str): name of the column
        order (int): order of the column
        data_type (DataType | str): data type of the column
        length (int | None): maximum length of the column
        vocabulary (str | None): code vocabulary of the column (e.g. ICD or SNOMED)
        primary_key (int | None): order of the column in the table primary key (if applicable)
        foreign_key (str | None): table.column identity of the foreign key (if applicable)
        enumerations (dict[str | int, str | int] | None): dictionary of code: value enumerations of the column
        description (str | None): description of the column
        datetime_format (str | None): identified datetime format of the column
    """

    def __init__(
        self,
        name: str,
        order: int,
        data_type: DataType | str,
        length: int | None = None,
        vocabulary: str | None = None,
        primary_key: int | None = None,
        foreign_key: str | None = None,
        enumerations: dict[str | int, str | int] | None = None,
        description: str | None = None,
        datetime_format: str | None = None,
    ):
        self.name = _normalise_name(name)
        self.order = int(order) if order is not None else None
        self.data_type: DataType = None
        self.length = int(length) if length is not None else None
        self.vocabulary = vocabulary
        self.primary_key = int(primary_key) if primary_key is not None else None
        self.foreign_key = foreign_key
        self.enumerations = enumerations or dict()
        self.description = description
        self.datetime_format = datetime_format
        self.set_data_type(data_type)
        self.check()

    # Magic
    def __repr__(self) -> str:
        data_type = (
            self.data_type.value
            if hasattr(self.data_type, "value")
            else str(self.data_type)
        )
        len_part = f"({self.length})" if self.length is not None else ""
        pk_part = (
            f", primary_key={self.primary_key!r}"
            if self.primary_key is not None
            else ""
        )
        datetime_format_part = (
            f", datetime_format={self.datetime_format!r}"
            if self.datetime_format
            else ""
        )
        return (
            f"Column(name={self.name!r}, order={self.order!r}, "
            + f"data_type='{data_type}{len_part}'{pk_part}{datetime_format_part})"
        )

    # Helpers
    def check(self) -> None:
        """
        Summary:
            Checks a Column object for errors.

        Raises:
            DataDictionaryError: if any errors are found in the Column object
        """
        errors = []
        errors.extend(_check_name(self.name, entity="column"))
        errors.extend(_check_order(self.order))
        errors.extend(_check_data_type(self.data_type, self.length))
        errors.extend(_check_primary_key(self.primary_key, self.data_type))

        if errors:
            raise DataDictionaryError(
                f"\nErrors in column {self.name!r}: {list_as_bullets(errors)}"
            )

    def set_data_type(self, data_type: DataType) -> None:
        self.data_type = (
            data_type if isinstance(data_type, DataType) else DataType.parse(data_type)
        )


class Table(list[Column]):
    """
    Summary:
        Represents a table in a data dictionary.

    Arguments:
        name (str): name of the table
        description (str | None): description of the table
        columns (list[Column] | None): list of columns in the table

    Raises:
        DataDictionaryError: if any errors are found in the Table object
    """

    def __init__(
        self,
        name: str,
        description: str | None = None,
        columns: list[Column] | None = None,
    ):
        super().__init__()
        self.name = _normalise_name(name)
        self.description = description
        for column in columns or []:
            self.add_column(column)
        self.check(instantiation=False if len(self) else True)

    def __repr__(self) -> str:
        cols_str = (
            "" if not self else f", {list_as_bullets(elements=[str(c) for c in self])}"
        )
        return f"Table(name={self.name!r}, description={self.description!r}{cols_str})"

    def __getitem__(self, key: int | str) -> Column:
        if isinstance(key, int):
            return super().__getitem__(key)
        target = _normalise_name(key)
        found = next((c for c in self if c.name == target), None)
        if not found:
            raise KeyError(f"Column {key!r} not found in table {self.name!r}.")
        return found

    def __get(self, name: str, default: Column | None = None) -> Column | None:
        target = _normalise_name(name)
        return next((c for c in self if c.name == target), default)

    # Getters
    def index_of(self, name: str) -> int | None:
        target = _normalise_name(name)
        for i, c in enumerate(self):
            if c.name == target:
                return i
        return None

    def get_column(self, column: str | int) -> Column:
        """
        Summary:
            Retrieves a column from the table by name or order.

        Args:
            column (str | int): name or order of the column to retrieve

        Returns:
            Column: the column with the specified name or order

        Raises:
            KeyError: if the specified column is not found in the table
        """
        if isinstance(column, str):
            col = self.__get(column)
            if col is None:
                raise KeyError(f"Column {column!r} not found in table {self.name!r}.")
            return col

        found = next((c for c in self if c.order == column), None)
        if not found:
            raise KeyError(
                f"Column with order {column!r} not found in table {self.name!r}."
            )
        return found

    def get_column_names(self) -> list[str]:
        """
        Summary:
            Retrieves a list of column names from the table.

        Returns:
            list[str]: a list of column names
        """
        return [c.name for c in self]

    def get_column_orders(self) -> list[int | None]:
        """
        Summary:
            Retrieves a list of column orders from the table.

        Returns:
            list[int | None]: a list of column orders
        """
        return [c.order for c in self]

    # Checkers
    def check(self, instantiation: bool = False) -> None:
        """
        Summary:
            Checks a Table object for errors.

        Arguments:
            instantiation (bool): whether this is an instantiation check or not. If
                not, additionally checks primary keys and orders.

        Raises:
            DataDictionaryError: if any errors are found in the Table object
        """
        errors = []
        errors.extend(_check_name(name=self.name, entity="table"))

        if not instantiation:
            errors.extend(self.__check_primary_keys())
            errors.extend(self.__check_orders())

        if errors:
            raise DataDictionaryError(
                f"\nErrors in table {self.name!r}: {list_as_bullets(errors)}"
            )

    def get_primary_keys(self) -> list[str]:
        """
        Summary:
            Retrieves a list of primary key column names from the table.

        Returns:
            list[str]: a list of primary key column names
        """
        primary_keys = []
        for column in self:
            if column.primary_key is not None:
                primary_keys.append(column.name)
        return primary_keys

    def __check_primary_keys(self) -> list[str]:
        errors: list[str] = []

        pk_cols = [c for c in self if c.primary_key is not None]
        if len(pk_cols) == 0:
            errors.append(
                "table has no Primary Key column(s). At least one is required"
            )
            return errors

        groups = defaultdict(list)
        for c in pk_cols:
            groups[c.primary_key].append(c.name)

        for ordinal, cols in groups.items():
            if len(cols) > 1:
                errors.append(
                    f"conflicting primary_key ordinal {ordinal}: used by columns {', '.join(repr(n) for n in cols)}."
                )

        return errors

    def __check_orders(self) -> list[str]:
        errors: list[str] = []
        groups = defaultdict(list)
        for c in self:
            groups[c.order].append(c.name)

        for order_val, cols in groups.items():
            if len(cols) > 1:
                errors.append(
                    f"conflicting order {order_val}: used by columns {', '.join(repr(n) for n in cols)}."
                )
        return errors

    # Manipulation
    def sort_columns(self) -> None:
        """
        Summary:
            Sorts the columns of the table in ascending order based on their order attribute.
        """
        self.sort(key=lambda c: c.order)

    def add_column(self, column: Column) -> None:
        """
        Summary:
            Adds a new column to the table.

        Arguments:
            column: the Column object to add to the table

        Raises:
            DataDictionaryError: if the column already exists, or if the order value is already in use by another column.
        """
        if not isinstance(column, Column):
            raise DataDictionaryError("Only Column objects can be added to a Table.")

        if column.name in self.get_column_names():
            conflict = self.get_column(column.name)
            raise DataDictionaryError(
                f"Column {column.name!r} already exists (order={conflict.order!r})"
            )

        if column.order in self.get_column_orders():
            conflict = self.get_column(column.order)
            raise DataDictionaryError(
                f"Order {column.order!r} already exists (name={conflict.name!r})"
            )

        if column.primary_key is not None:
            pk_conflict = next(
                (c for c in self if c.primary_key == column.primary_key), None
            )
            if pk_conflict is not None:
                raise DataDictionaryError(
                    f"Primary key ordinal {column.primary_key} for {column.name!r} "
                    f"conflicts with existing column {pk_conflict.name!r}."
                )

        super().append(column)
        self.sort_columns()

    def remove_column(self, column: str | int) -> None:
        """
        Summary:
            Removes a column from the table.

        Arguments:
            column: the column string or order to remove

        Raises:
            DataDictionaryError: if the column does not exist
        """
        if isinstance(column, str):
            name = self.get_column(column).name
        else:
            name = self.get_column(column).name  # by order
        remaining = [c for c in self if c.name != name]
        self.clear()
        super().extend(remaining)

    def set_primary_keys(self, primary_keys: list[str | int]) -> None:
        """
        Summary:
            Sets primary keys for the table.

        Arguments:
            primary_keys: list of column names or orders to set as primary keys

        Raises:
            DataDictionaryError: if primary keys were not provided (empty list)
        """
        if not primary_keys:
            raise DataDictionaryError(
                f"Primary keys for table {self.name!r} were not provided (empty list)."
            )

        # Clear existing PKs
        for col in self:
            col.primary_key = None

        # Resolve and dedupe
        resolved: list[Column] = []
        seen: set[str] = set()
        for key in primary_keys:
            col = self.get_column(key)
            if col.name in seen:
                raise DataDictionaryError(
                    f"Duplicate column {col.name!r} provided for table {self.name!r}."
                )
            seen.add(col.name)
            resolved.append(col)

        # Assign ordinals 1..N
        for ordinal, col in enumerate(resolved, start=1):
            col.primary_key = ordinal
            # Column.check() enforces PK validity for the column's data_type
            col.check()

        # Table-level validation (presence, unique ordinals)
        self.check()


class Dictionary(list[Table]):
    """A collection of tables and metadata describing a dataset, against which records
    can be validated.

    Attributes:
        name (str | None): Name of the dataset or project
        organisations (str | None): Organisations collaborating on the dataset
        version (str | None): Version number of the dataset (e.g. v1.0)
        version_notes (str | None): Notes about the dataset version (e.g. changes made)
        inclusion_criteria (str | None): Cohort inclusion criteria
        exclusion_criteria (str | None): Cohort exclusion criteria
        imported (bool): Whether the dictionary has been imported from an external source (e.g. Excel)
    """

    def __init__(
        self,
        name: str | None = None,
        tables: list[Table] | None = None,
        organisations: str | None = None,
        version: str | None = None,
        version_notes: str | None = None,
        inclusion_criteria: str | None = None,
        exclusion_criteria: str | None = None,
        imported: bool = False,
    ):
        super().__init__()
        self.name = name
        for t in tables or []:
            self.add_table(t)
        self.organisations = organisations
        self.version = version
        self.version_notes = version_notes
        self.inclusion_criteria = inclusion_criteria
        self.exclusion_criteria = exclusion_criteria
        self.imported = imported

    # Properties
    @property
    def table_count(self) -> int:
        return len(self)

    @property
    def column_count(self) -> int:
        return 0 if not self.table_count else sum(len(table) for table in self)

    # Magic
    def __repr__(self) -> str:
        tables = list_as_bullets(elements=[str(t) for t in self], bullet="\n- ")
        return f"Dictionary(name={self.name!r}, imported={self.imported!r}, {tables})"

    def __getitem__(self, key: int | str) -> Table:
        if isinstance(key, int):
            return super().__getitem__(key)
        target = _normalise_name(key)
        found = next((t for t in self if t.name == target), None)
        if not found:
            raise KeyError(f"Table {key!r} not found in Dictionary.")
        return found

    # Getters
    def __get(self, name: str, default: Table | None = None) -> Table | None:
        target = _normalise_name(name)
        return next((t for t in self if t.name == target), default)

    def index_of(self, name: str) -> int | None:
        target = _normalise_name(name)
        for i, t in enumerate(self):
            if t.name == target:
                return i
        return None

    def get_table_names(self) -> list[str]:
        """
        Summary:
            Retrieves a list of table names from the dictionary.

        Returns:
            list[str]: A list of table names.
        """
        return [t.name for t in self]

    def get_table(self, table: str) -> Table:
        """
        Summary:
            Gets a table from the dictionary by name.

        Arguments:
            table (str): The name of the table to be retrieved.

        Returns:
            Table: The retrieved table.

        Raises:
            KeyError: If the table is not found in the dictionary.
        """
        target = _normalise_name(table)
        found = next((t for t in self if t.name == target), None)

        if not found:
            raise KeyError(f"Table {table!r} not found in Dictionary.")

        return found

    # Manipulation
    def add_table(self, table: Table) -> None:
        """
        Summary:
            Adds a table to the dictionary.

        Arguments:
            table (Table): The table to be added.

        Raises:
            DataDictionaryError: If the table already exists in the dictionary.
        """
        if not isinstance(table, Table):
            raise DataDictionaryError(
                "Only Table objects can be added to a Dictionary."
            )
        if table.name in self.get_table_names():
            raise DataDictionaryError(f"Table {table.name!r} already exists.")
        super().append(table)

    def remove_table(self, table: str) -> None:
        """
        Summary:
            Removes the specified table from the dictionary.

        Arguments:
            table (str): The name of the table to be removed.

        Raises:
            DataDictionaryError: If the table does not exist in the dictionary.
        """
        name = self.get_table(table).name
        remaining = [t for t in self if t.name != name]
        self.clear()
        super().extend(remaining)

    def set_primary_keys(self, primary_keys: dict[str, list[str | int]]) -> None:
        """
        Summary:
            Sets the primary keys for each table in the dictionary.

        Arguments:
            primary_keys (dict[str, list[str | int]]): A dictionary mapping table names to column names or orders.

        Raises:
            DataDictionaryError: If any tables or columns have invalid names or types, or if any tables or columns have duplicate names.
        """
        for table_name, keys in (primary_keys or {}).items():
            self.get_table(table_name).set_primary_keys(keys)

    # Helpers
    def check(self) -> None:
        """
        Summary:
            Validates the integrity of the dictionary.

        Raises:
            DataDictionaryError: If any tables or columns have invalid names or
                types, or if any tables or columns have duplicate names.
        """
        for table in self:
            table.check()

        for table in self:
            for column in table:
                column.check()

    # Export
    def export_dictionary(
        self,
        directory: Path | str,
        filename: str | None = None,
        overwrite: bool = False,
        debug: bool = False,
        _template_path: Path | str | None = None,
    ):
        from valediction.dictionary.exporting import (
            export_dictionary,  # Avoid Circulars
        )

        return export_dictionary(
            dictionary=self,
            directory=directory,
            filename=filename,
            overwrite=overwrite,
            debug=debug,
            _template_path=_template_path,
        )
