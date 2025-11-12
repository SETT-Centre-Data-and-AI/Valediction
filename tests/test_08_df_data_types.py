import pytest
from pandas import DataFrame

from valediction import Dataset, demo
from valediction.data_types.data_types import DataType
from valediction.dictionary.model import Table

EXPECTED_DTYPES = {
    DataType.TEXT: "string",
    DataType.INTEGER: "Int64",
    DataType.FLOAT: "Float64",
    DataType.DATE: "datetime64[ns]",
    DataType.DATETIME: "datetime64[ns]",
}


# Helpers
def get_dtype(data_type: DataType) -> str:
    return EXPECTED_DTYPES[data_type]


def get_dataset() -> Dataset:
    dataset = Dataset.create_from(demo.DEMO_DATA)
    dataset.import_dictionary(demo.DEMO_DICTIONARY)
    return dataset


def check_imported_as_strings(df: DataFrame, table_dictionary: Table) -> None:
    for col, dtype in df.dtypes.items():
        assert str(dtype) == "string", (
            f"Column '{col}' imported as '{dtype}', not 'string'"
        )


def check_column_dtypes(df: DataFrame, table_dictionary: Table) -> None:
    for col, dtype in df.dtypes.items():
        data_type = table_dictionary.get_column(col).data_type
        assert str(dtype) == get_dtype(data_type), (
            f"Column '{col}' is '{dtype}', not '{get_dtype(data_type)}'"
        )


# Tests
def test_imports_as_strings() -> None:
    dataset = get_dataset()
    dataset.import_data()

    for item in dataset:
        check_imported_as_strings(item.data, item.table_dictionary)


def test_types_applied_import_then_validate() -> None:
    dataset = get_dataset()
    dataset.import_data()
    dataset.validate()

    for item in dataset:
        check_column_dtypes(item.data, item.table_dictionary)


def test_types_applied_validate_then_import() -> None:
    dataset = get_dataset()
    dataset.validate()
    dataset.import_data()

    for item in dataset:
        check_column_dtypes(item.data, item.table_dictionary)


def test_types_string_on_chunking() -> None:
    dataset = get_dataset()
    for item in dataset:
        for chunk in item.iterate_data_chunks():
            df = chunk.df
            check_imported_as_strings(df, item.table_dictionary)


def test_types_applied_on_validate_then_chunking() -> None:
    dataset = get_dataset()
    dataset.validate()
    for item in dataset:
        for chunk in item.iterate_data_chunks():
            df = chunk.df
            check_column_dtypes(df, item.table_dictionary)


def test_types_applied_on_chunking_when_imported() -> None:
    dataset = get_dataset()
    dataset.validate()
    dataset.import_data()
    for item in dataset:
        for chunk in item.iterate_data_chunks():
            df = chunk.df
            check_column_dtypes(df, item.table_dictionary)


if __name__ == "__main__":
    pytest.main([__file__])
