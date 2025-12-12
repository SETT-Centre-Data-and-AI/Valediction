from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable, Iterator

from pandas import DataFrame

from valediction.datasets.datasets_helpers import DataLike
from valediction.dictionary.generation import Generator
from valediction.dictionary.importing import import_dictionary
from valediction.dictionary.model import Dictionary, Table
from valediction.exceptions import DataDictionaryImportError, DataIntegrityError
from valediction.io.csv_readers import (
    FrameChunk,
    iter_csv_chunks,
    read_csv_all,
    read_csv_headers,
)
from valediction.support import (
    _get_runtime_string,
    _normalise_name,
    list_as_bullets,
    print_bold_red,
    print_red,
)
from valediction.validation.helpers import apply_data_types
from valediction.validation.issues import Issues
from valediction.validation.validation import Validator


@dataclass()
class DatasetItem:
    """
    Summary:
        Represents a single table binding for validation.

    Attributes:
        name (str): table name
        data (Path | DataFrame): DataFrame or path to csv
        validated (bool): whether the table has been successfully validated
        table_dictionary (Table | None): dictionary Table object for the DatasetItem
        validator (Validator): validator object
        issues (Issues): contains validation issues/deviations from the dictionary
        validation_runtimes (dict[str, str]): validation runtime
        dictionary_runtimes (dict[str, str]): dictionary generation runtime

    Raises:
        DataDictionaryImportError: if there is an issue with importing the dictionary
        DataIntegrityError: if there is an issue with the integrity of the data
    """

    name: str
    data: Path | DataFrame
    validated: bool = False
    table_dictionary: Table | None = None
    validator: Validator = None
    issues: Issues = None
    _validation_runtimes: dict[str, timedelta] = None
    _dictionary_runtimes: dict[str, timedelta] = None
    _padding: int = 0

    def __post_init__(self):
        object.__setattr__(self, "issues", Issues())

    # Properties
    @property
    def validation_runtimes(self) -> dict[str, str]:
        if not self._validation_runtimes:
            return {}

        return {
            step: _get_runtime_string(time_delta)
            for step, time_delta in self._validation_runtimes.items()
        }

    @property
    def dictionary_runtimes(self) -> dict[str, str]:
        if not self._dictionary_runtimes:
            return {}

        return {
            step: _get_runtime_string(time_delta)
            for step, time_delta in self._dictionary_runtimes.items()
        }

    @property
    def is_dataframe(self) -> bool:
        return isinstance(self.data, DataFrame)

    @property
    def is_path(self) -> bool:
        return isinstance(self.data, Path)

    @property
    def column_count(self) -> int:
        if isinstance(self.data, DataFrame):
            return self.data.shape[1]
        else:
            return read_csv_headers(path=self.data).shape[1]

    @property
    def primary_keys(self) -> list[str]:
        if not self.table_dictionary:
            raise DataDictionaryImportError(
                "No dictionary attached to table - please import_dictionary() or generate_dictionary() first"
            )

        return self.table_dictionary.get_primary_keys()

    @property
    def headers(self) -> list[str]:
        if self.is_dataframe:
            return list(self.data.columns)
        elif self.is_path:
            return list(read_csv_headers(path=self.data).columns)
        else:
            raise TypeError("self.data must be a DataFrame or str/Path to .csv")

    # Magic
    def __repr__(self) -> str:
        if isinstance(self.data, DataFrame):
            shape = f"{self.data.shape[0]}x{self.data.shape[1]}"
            data_repr = f"DataFrame[{shape}]"
        elif isinstance(self.data, Path):
            data_repr = f"Path('{self.data.name}')"
        else:
            data_repr = repr(self.data)
        return (
            f"DatasetItem(name={self.name!r}, data={data_repr}, "
            f"validated={self.validated})"
        )

    # Validation
    def validate(
        self,
        chunk_size: int | None = 10_000_000,
        feedback: bool = True,
    ) -> None:
        """
        Summary:
            Validates the dataset item against the dictionary.
            Warns if there are issues with the integrity of the data.

        Arguments:
            chunk_size (int | None): Size of chunks for validating data to optimise RAM usage,
                if reading from CSV (default: 10_000_000)
            feedback (bool): Provide user feedback on progress (default: True)

        Raises:
            DataDictionaryImportError: if there is an issue with importing the dictionary
        """
        self.__check_dictionary()
        validator = Validator(
            dataset_item=self,
            table_dictionary=self.table_dictionary,
            chunk_size=chunk_size,
            feedback=feedback,
            _padding=self._padding,
        )

        object.__setattr__(self, "validator", validator)
        try:
            validator.validate()
            object.__setattr__(self, "validated", True)
            object.__setattr__(self, "issues", Issues())
            if self.is_dataframe:
                self.apply_dictionary()

        # Issues detected
        except DataIntegrityError:
            object.__setattr__(self, "validated", False)
            object.__setattr__(self, "issues", validator.issues)

        # No Issues
        else:
            object.__setattr__(self, "validated", True)

        finally:
            object.__setattr__(self, "_validation_runtimes", validator._runtimes)

            # Warn Issues
            try:
                self.check()
            except DataIntegrityError:
                pass

    def check(self) -> bool:
        """
        Summary:
            Check the validity of the DatasetItem.

        Raises:
            DataIntegrityError: If there is an issue with the integrity of the data, either because:
                - the DatasetItem is not yet validated
                - there are issues with the integrity of the data
        """
        error = (
            f"Issues detected in {self.name}. Issues:\n{self.issues}"
            if len(self.issues) > 0
            else "DatasetItem not yet validated"
            if not self.validated
            else ""
        )
        if error:
            print_bold_red(f"WARNING: Issues detected in {self.name}.")
            print_red(f"{self.issues}")
            raise DataIntegrityError(error)
        else:
            return True

    def apply_dictionary(self):
        """
        Summary:
            Apply a validated Data Dictionary to a validated DatasetItem.

        Raises:
            DataDictionaryImportError: if no Data Dictionary has been imported or generated and attached to the table
            DataIntegrityError: if the data has not been validated before attempting to apply the dictionary
        """
        if not self.table_dictionary:
            raise DataDictionaryImportError(
                "No Data Dictionary imported or generated and attached to table. "
                + "Please first run Dataset.import_dictionary() or Dataset.generate_dictionary() "
                + " and then Dataset.validate()"
            )

        if not self.validated:
            raise DataIntegrityError(
                "Cannot apply Data Dictionary to unvalidated data. "
                + "Please first run DataSet.validate() on the table."
            )

        if self.is_path:
            self.import_data()

        object.__setattr__(
            self, "data", apply_data_types(self.data, self.table_dictionary)
        )

    # Data Import
    def import_data(self):
        """
        Summary:
            Import the data associated with this DatasetItem into memory.

        Raises:
            DataIntegrityError: if there is an issue with the integrity of the data
        """
        if self.is_dataframe:
            print(f"DatasetItem '{self.name}' already imported")
            return

        else:
            object.__setattr__(self, "data", read_csv_all(self.data).df)
            if self.table_dictionary and self.validated:
                self.apply_dictionary()

    def iterate_data_chunks(self, chunk_size: int = 10_000_000) -> Iterator[FrameChunk]:
        """
        Summary:
            Yields data in chunks. If `data` is a DataFrame, yields the whole DataFrame once within
            a FrameChunk. If the Dataset is validated, dtypes will be applied to the DataFrame.
            If not, will warn and return as strings types.

        Args:
            chunk_size (int, optional): chunk_size (int | None): Size of chunks for reading data to optimise RAM usage,
                if reading from CSV (default: 10_000_000)

        Yields:
            Iterator[FrameChunk]: Iterator of FrameChunks, with each chunk containing a DataFrame as `chunk.df`
        """
        if not self.validated:
            print_bold_red("WARNING: ", end="")
            print_red(
                f"DatasetItem '{self.name}' has not been validated. "
                + "All data will be yielded with string dtypes."
            )
        if self.is_path:
            for chunk in iter_csv_chunks(path=self.data, chunk_size=chunk_size):
                if self.validated:
                    df = apply_data_types(chunk.df, self.table_dictionary)
                    chunk.update_df(df)
                yield chunk

        if self.is_dataframe:
            n = len(self.data)
            # apply_data_types() will already have been applied if validated
            yield FrameChunk(
                df=self.data,
                start=0,
                end=(n - 1) if n else 0,
                total_size=None,
                file_pos=None,
                bytes_read=None,
                chunk_index=1,
                total_bytes_read=None,
                total_chunks_seen=1,
            )
            return

    # Data Export
    def export_data(
        self,
        directory: Path | str,
        overwrite: bool = False,
        enforce_validation: bool = True,
    ):
        """Export DatasetItem data to csv, if imported.

        Args:
            directory (Path | str): Directory to export csv file.
            overwrite (bool, optional): Overwrite existing file on conflict. Defaults to False.
            enforce_validation (bool, optional): Raise error if unvalidated. Defaults to True.

        Raises:
            ValueError: If unimported, unvalidated and enforced, or exists without overwrite
        """
        if not isinstance(directory, (Path, str)):
            raise TypeError(f"directory must be a Path/str, not {type(directory)}")

        if self.is_path:
            raise ValueError(
                f"Data '{self.name}' is not imported. Run self.import_data()"
            )

        if not self.validated:
            if enforce_validation:
                raise ValueError(
                    f"DatasetItem '{self.name}' has not been validated. "
                    + "Please first run self.validate() on the DatasetItem or Dataset."
                )

        directory = Path(directory)
        filename = f"{self.name}.csv"

        if not directory.exists():
            directory.mkdir(parents=True)

        out_path = directory / filename
        if out_path.exists() and not overwrite:
            raise ValueError(f"File exists and overwrite=False: {out_path}")

        self.data.to_csv(out_path, index=False)

    # Helpers
    def _attach_table_dictionary(self, table_dictionary: Table):
        object.__setattr__(self, "table_dictionary", table_dictionary)
        object.__setattr__(self, "validated", False)

    def _set_padding(self, padding: int):
        object.__setattr__(self, "_padding", padding)

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


class Dataset(list[DatasetItem]):
    """
    Summary:
        A list-like container of DatasetItem with helpful name-based accessors and
        creators. Also holds an optional Dictionary and can generate one from
        the current items.

    Arguments:
        items (Iterable[DatasetItem] | None): An iterable of DatasetItem objects.
    """

    # Properties
    @property
    def validated(self) -> bool:
        return all([item.validated for item in self])

    # Magic
    def __init__(self, items: Iterable[DatasetItem] | None = None) -> None:
        super().__init__(items or [])
        self.dictionary: Dictionary | None = None
        self.issues: Issues = None

    def __repr__(self) -> str:
        base = f"Dataset(len={len(self)}, dictionary_loaded={self._dd_loaded()}"
        items_str = (
            ")"
            if not len(self)
            else f", {list_as_bullets(elements=[str(d) for d in self])}\n)"
        )

        return f"{base}{items_str}"

    # Creation
    @classmethod
    def create_from(
        cls,
        dataset: Path | str | dict[str, DataFrame],
        *,
        overwrite: bool = False,
    ) -> Dataset:
        """Build a Dataset from a path (file/dir) or dictionary of {name: DataFrame}."""
        if not isinstance(dataset, (Path, str, dict)):
            raise TypeError(
                f"dataset must be a Path/str or dict[str, DataFrame], not {type(dataset)}"
            )

        ds = cls()

        # Path-like input
        if isinstance(dataset, (Path, str)):
            items = cls._items_from_pathlike(Path(dataset))
            if len(items) == 1:
                ds.add(items[0], overwrite=overwrite)
            else:
                ds.extend_add(items, overwrite=overwrite)

        # Iterable input
        else:
            items = [
                cls._make_item(name=name, data=data) for name, data in dataset.items()
            ]
            ds.extend_add(items, overwrite=overwrite)

        # Set Padding
        max_length = max(len(item.name) for item in ds)
        for item in ds:
            padding = max_length - len(item.name)
            item._set_padding(padding)
        return ds

    # Getters
    def get(self, name: str, default: DatasetItem | None = None) -> DatasetItem | None:
        name_key = name.strip()
        for item in self:
            if item.name.lower() == name_key.lower():
                return item
        return default

    def index_of(self, name: str) -> int | None:
        name_key = name.strip()
        for i, item in enumerate(self):
            if item.name == name_key:
                return i
        return None

    # Manipulation
    def add(self, item: DatasetItem, *, overwrite: bool = False) -> None:
        """
        Summary:
            Add a new DatasetItem to the end of the Dataset, optionally
            overwriting any existing item with the same name.

        Arguments:
            item (DatasetItem): The DatasetItem to be added.
            overwrite (bool): Whether to overwrite any existing item with the same name.
                Defaults to False.

        Raises:
            ValueError: If an item with the same name already exists and overwrite is False.
        """
        existing_index = self.index_of(item.name)
        if existing_index is not None and not overwrite:
            raise ValueError(
                f"Item with name '{item.name}' already exists. Use overwrite=True to replace."
            )
        if existing_index is None:
            self.append(item)
        else:
            self[existing_index] = item

    def extend_add(
        self, items: Iterable[DatasetItem], *, overwrite: bool = False
    ) -> None:
        """
        Summary:
            Extend the Dataset by adding multiple DatasetItems.

        Arguments:
            items (Iterable[DatasetItem]): An iterable of DatasetItems to be added.
            overwrite (bool): Whether to overwrite any existing item with the same name.
                Defaults to False.

        Raises:
            ValueError: If an item with the same name already exists and overwrite is False.
        """
        for it in items:
            self.add(it, overwrite=overwrite)

    # Data Dictionary
    def import_dictionary(self, dictionary: Dictionary | Path | str) -> None:
        """
        Summary:
            Attach a dictionary to the Dataset.

        Arguments:
            dictionary (Dictionary | Path | str): A dictionary to be attached, either as a Dictionary object
                or a Path/str filepath to compatible dictionary .xlsx file.

        Raises:
            TypeError: If the dictionary is not a Dictionary instance or a Path/str to an importable file.
        """
        if isinstance(dictionary, Dictionary):
            self.dictionary = dictionary
        elif isinstance(dictionary, (Path, str)):
            path = Path(dictionary)
            self.dictionary = import_dictionary(path)
        else:
            raise TypeError(
                "dictionary must be a Dictionary instance or a Path/str to an importable file."
            )

        self._attach_table_dictionaries()

    # Data Dictionary
    def export_dictionary(
        self,
        directory: Path | str,
        filename: str | None = None,
        overwrite: bool = False,
        debug: bool = False,
        _template_path: Path | str | None = None,
    ):
        """
        Summary:
            Export a data dictionary to an Excel file.

        Arguments:
            directory (Path | str): The directory to export to.
            filename (str | None): The filename to export to (default is None).
            overwrite (bool): Whether to overwrite existing file (default is False).
            debug (bool): Whether to print debug information (default is False).
            _template_path (Path | str | None): The path to the template data dictionary
                (default is None; changing not advised).

        Returns:
            None

        Raises:
            FileNotFoundError: If the directory specified by directory does not exist.
            ValueError: If the file specified by filename already exists and overwrite is False.
        """
        if getattr(self, "dictionary", None) is None:
            raise ValueError("No Dictionary attached to this Dataset.")
        from valediction.dictionary.exporting import (
            export_dictionary,  # Avoid circular import
        )

        return export_dictionary(
            dictionary=self.dictionary,  # type: ignore[arg-type]
            directory=directory,
            filename=filename,
            overwrite=overwrite,
            debug=debug,
            _template_path=_template_path,
        )

    def generate_dictionary(
        self,
        dictionary_name: str | None = None,
        primary_keys: dict[str, list[str | int]] | None = None,
        feedback: bool = True,
        debug: bool = False,
        chunk_size: int | None = 10_000_000,
        sample_rows: int | None = None,
    ) -> Dictionary:
        """
        Summary:
            Generate a dictionary from a Dataset.

        Arguments:
            dictionary_name (str | None): The name of the dictionary to generate.
                If None, will not be set.
            primary_keys (dict[str, list[str | int]] | None): A dictionary of primary keys
                to set on the generated dictionary. If None, will not be set.
            feedback (bool): Provide user feedback on progress (default: True)
            debug (bool): Enable debug mode, providing full log of data type inference and
                reasoning (default: False)
            chunk_size (int | None): Size of chunks for reading data to optimise RAM usage,
                if reading from CSV (default: 10_000_000)
            sample_rows (int | None): Number of rows to sample for data type inference. Note:
                this overrides `chunk_size` and reads in a single chunk (default: None)

        Returns:
            Dictionary: The generated dictionary.
        """
        generator = Generator(
            feedback=feedback,
            debug=debug,
            chunk_size=chunk_size,
            sample_rows=sample_rows,
        )
        dictionary = generator.generate_dictionary(
            self,
            dictionary_name=dictionary_name,
            primary_keys=primary_keys,
        )
        self.dictionary = dictionary
        self._attach_table_dictionaries()
        return dictionary

    # Data
    def import_data(
        self,
        name: str | None = None,
    ) -> None:
        """
        Summary:
            Import data from CSV files into the Dataset.

        Arguments:
            name (str | None): The name of the table to import data into. If None, all tables are imported.

        Raises:
            FileNotFoundError: If the file specified by name does not exist.
        """
        if name:
            self[name].import_data()

        else:
            for item in self:
                if item.is_path:
                    item.import_data()

    def export_data(
        self,
        directory: Path | str,
        overwrite: bool = False,
        enforce_validation: bool = True,
    ):
        """Export items from Dataset data to csv, if imported. Unimported items are
        skipped. Unvalidated items are skipped if enforce_validation is True.

        Args:
            directory (Path | str): Directory to export csv files.
            overwrite (bool, optional): Overwrite existing files on conflict. Defaults to False.
            enforce_validation (bool, optional): Raise error if unvalidated. Defaults to True.

        Raises:
            ValueError: If files exists without overwrite=True.
        """
        if not isinstance(directory, (Path, str)):
            raise TypeError(f"directory must be a Path/str, not {type(directory)}")
        print("Exporting data...")
        # Check for issues
        unimported_items = [item for item in self if item.is_path]
        unvalidated_items = [
            item for item in self if item.is_dataframe and not item.validated
        ]

        if unimported_items:
            print_bold_red("WARNING: Skipping unimported tables: ", end="")
            print_red(list_as_bullets([item.name for item in unimported_items]))

        if unvalidated_items and enforce_validation:
            print_bold_red("WARNING: Skipping unvalidated tables: ", end="")
            print_red(list_as_bullets([item.name for item in unvalidated_items]))

        # Set exportable
        exportable: list[DatasetItem] = []
        for item in self:
            if item.is_dataframe:
                if item.validated or not enforce_validation:
                    exportable.append(item)

        directory = Path(directory)
        filenames = [directory / f"{item.name}.csv" for item in exportable]

        # Check for conflicts and overwrite config
        conflicts = [str(filename) for filename in filenames if filename.exists()]
        if conflicts and not overwrite:
            raise ValueError(
                f"File exists and overwrite=False: {list_as_bullets(conflicts)}"
            )

        # Export
        for item in exportable:
            print(f" - exporting '{item.name}'")
            item.export_data(directory, overwrite=overwrite, enforce_validation=False)

        print(f"Export complete ({len(exportable)} tables)")

    def apply_dictionary(self, name: str | None = None) -> None:
        """
        Summary:
            Apply a dictionary to a Dataset.

        Arguments:
            name (str | None): The name of the table to apply the dictionary to. If None, all tables are applied.

        Returns:
            None

        Raises:
            ValueError: If the Dataset does not contain a dictionary.
        """
        if name:
            self[name].apply_dictionary()

        else:
            for item in self:
                item.apply_dictionary()

    # Validation
    def validate(
        self,
        chunk_size: int | None = 10_000_000,
        feedback: bool = True,
    ) -> None:
        """
        Summary:
            Validate data in the Dataset against the dictionary.

        Arguments:
            chunk_size (int): Size of chunks for validating data to optimise RAM usage.
            feedback (bool): Provide user feedback on progress (default: True)

        Returns:
            None

        Raises:
            DataIntegrityError: If there is an issue with the integrity of the data
            DataDictionaryImportError: If there is an issue with importing the dictionary
        """
        if feedback:
            print(f"Validating {len(self)} tables")
        self.__check_dictionary()
        for item in self:
            try:
                item.validate(
                    chunk_size=chunk_size,
                    feedback=feedback,
                )
            except DataIntegrityError:
                pass

        self.__reattach_issues()

        # Report Issues
        try:
            self.check(readout=True)
        except DataIntegrityError:
            pass

        if feedback:
            print("\n", end="")

    def __reattach_issues(self) -> None:
        self.issues = Issues()
        for item in self:
            self.issues.extend(item.issues)

    def __items_with_issues(self) -> list[str]:
        items_with_issues = [item.name for item in self if len(item.issues) > 0]
        string = (
            ("" + ",".join(items_with_issues) + "")
            if len(items_with_issues) > 0
            else ""
        )
        return string

    def check(self, readout: bool = False) -> bool:
        """
        Summary:
            Check the validity of the Dataset.

        Raises:
            DataIntegrityError: If there is an issue with the integrity of the data, either because:
                - the Dataset is not yet validated
                - there are issues with the integrity of the data
        """
        error = (
            f"WARNING: Unvalidated tables or issues detected in {self.__items_with_issues()}:"
            if len(self.issues) > 0
            else "Dataset not yet validated"
            if not self.validated
            else ""
        )
        if error:
            if readout:
                print_bold_red(f"\n{error}")
                print_red(self.issues)
            raise DataIntegrityError(f"{error}\n{self.issues}")
        else:
            return True

    # Creation Helpers
    @staticmethod
    def _make_item(
        name: str | None,
        data: DataLike,
    ) -> DatasetItem:
        """Normalise a (data, name) double into a DatasetItem."""
        if isinstance(data, (str, Path)):
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            if path.suffix.lower() != ".csv":
                raise ValueError(f"Only .csv supported right now, got: {path}")
            resolved_name = _normalise_name(name or path.stem)
            return DatasetItem(name=resolved_name, data=path.resolve())

        if isinstance(data, DataFrame):
            if not name:
                raise ValueError("When providing a DataFrame, 'name' is required.")
            resolved_name = _normalise_name(name)
            return DatasetItem(name=resolved_name, data=data)

        raise TypeError("data must be a Path/str to .csv or a pandas DataFrame.")

    @staticmethod
    def _items_from_pathlike(p: Path) -> list[DatasetItem]:
        """Expand a file/dir path into DatasetItems (non-recursive for dirs)."""
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {p}")

        if p.is_file():
            if p.suffix.lower() != ".csv":
                raise ValueError(f"Expected a .csv file, got: {p.suffix} ({p})")
            return [DatasetItem(name=_normalise_name(p.stem), data=p.resolve())]

        if p.is_dir():
            return [
                DatasetItem(
                    name=_normalise_name(csv_path.stem), data=csv_path.resolve()
                )
                for csv_path in p.glob("*.csv")
            ]

        raise ValueError(f"Unsupported path type: {p}")

    # Validation Helpers
    def __check_dictionary(self):
        if self.dictionary is None or not isinstance(self.dictionary, Dictionary):
            raise DataDictionaryImportError(
                "Data Dictionary not yet imported or generated. "
                + "Validation must first have a Data Dictionary. "
                + "Please first run DataSet.import_dictionary(), including `primary_keys`."
            )

        self.dictionary.check()

    # Other Helpers
    def __getitem__(self, key: int | str) -> DatasetItem:
        if isinstance(key, int):
            return super().__getitem__(key)
        found = self.get(key)
        if found is None:
            raise KeyError(f"No DatasetItem with name '{key}'.")
        return found

    def _dd_loaded(self):
        return self.dictionary is not None

    def _attach_table_dictionaries(self):
        for dataset_item in self:
            table_name = dataset_item.name
            table_dictionary = self.dictionary.get_table(table_name)
            if not table_dictionary:
                raise DataDictionaryImportError(
                    f"No dictionary table found for '{table_name}'"
                )

            dataset_item._attach_table_dictionary(table_dictionary)
