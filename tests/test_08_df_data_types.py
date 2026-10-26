import pytest
from pandas import DataFrame

from valediction import Dataset, demo
from valediction.data_types.data_types import DataType
from valediction.dictionary.model import Column, Dictionary, Table

EXPECTED_DTYPES = {
    DataType.TEXT: "string",
    DataType.INTEGER: "Int64",
    DataType.FLOAT: "Float64",
    DataType.DATE: "datetime64[ns]",
    DataType.TIMESTAMP: "datetime64[ns]",
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
    dataset.validate(feedback=False)

    for item in dataset:
        check_column_dtypes(item.data, item.table_dictionary)


def test_types_applied_validate_then_import() -> None:
    dataset = get_dataset()
    dataset.validate(feedback=False)
    dataset.import_data()

    for item in dataset:
        check_column_dtypes(item.data, item.table_dictionary)


def test_validation_can_defer_dataframe_dtypes_until_iteration() -> None:
    dataset = get_dataset()
    dataset.import_data()

    dataset.validate(feedback=False, apply_dtypes=False)

    item = dataset[0]
    assert item.validated
    assert not item._dtypes_applied
    for chunk in item.iterate_data_chunks(warn=False):
        check_column_dtypes(chunk.df, item.table_dictionary)
    assert item._dtypes_applied


def test_dataset_item_validation_can_defer_dictionary_dtypes() -> None:
    dataset = get_dataset()
    dataset.import_data()
    item = dataset[0]

    item.validate(feedback=False, apply_dtypes=False)

    assert item.validated
    assert not item._dtypes_applied
    check_imported_as_strings(item.data, item.table_dictionary)

    item.apply_dictionary()

    check_column_dtypes(item.data, item.table_dictionary)


def test_revalidation_defers_replacement_dictionary_dtypes_until_iteration() -> None:
    dataset = Dataset.create_from(
        {"TABLE": DataFrame({"ID": ["1", "2"], "VALUE": ["2", "3"]}, dtype="string")}
    )
    dataset.import_dictionary(
        Dictionary(
            name="Integer values",
            tables=[
                Table(
                    name="TABLE",
                    columns=[
                        Column("ID", 1, "int", primary_key=1),
                        Column("VALUE", 2, "int"),
                    ],
                )
            ],
        )
    )
    dataset.validate(feedback=False)
    item = dataset["TABLE"]
    assert str(item.data["VALUE"].dtype) == "Int64"
    assert item._dtypes_applied

    dataset.import_dictionary(
        Dictionary(
            name="Float values",
            tables=[
                Table(
                    name="TABLE",
                    columns=[
                        Column("ID", 1, "int", primary_key=1),
                        Column("VALUE", 2, "float"),
                    ],
                )
            ],
        )
    )
    item.validate(feedback=False, apply_dtypes=False)

    assert item.validated
    assert not item._dtypes_applied
    assert str(item.data["VALUE"].dtype) == "Int64"

    chunk = next(item.iterate_data_chunks(warn=False))

    assert str(chunk.df["VALUE"].dtype) == "Float64"
    assert item._dtypes_applied


def test_types_string_on_chunking() -> None:
    dataset = get_dataset()
    for item in dataset:
        for chunk in item.iterate_data_chunks():
            df = chunk.df
            check_imported_as_strings(df, item.table_dictionary)


def test_chunking_can_suppress_unvalidated_warning(capsys) -> None:
    item = get_dataset()[0]

    chunk = next(item.iterate_data_chunks(warn=False))

    assert not chunk.df.empty
    assert "has not been validated" not in capsys.readouterr().out


def test_validated_chunking_can_skip_dtype_application(tmp_path) -> None:
    csv_path = tmp_path / "TABLE.csv"
    DataFrame({"ID": [1, 2], "VALUE": ["null", "2"]}).to_csv(csv_path, index=False)
    dataset = Dataset.create_from(csv_path)
    dataset.import_dictionary(
        Dictionary(
            name="Deferred chunk dtypes",
            tables=[
                Table(
                    name="TABLE",
                    columns=[
                        Column("ID", 1, "int", primary_key=1),
                        Column("VALUE", 2, "int"),
                    ],
                )
            ],
        )
    )
    dataset.validate(feedback=False)
    item = dataset["TABLE"]

    chunk = next(item.iterate_data_chunks(apply_dtypes=False))

    assert chunk.df["VALUE"].isna().tolist() == [True, False]
    assert str(chunk.df["VALUE"].dtype) == "string"


def test_types_applied_on_validate_then_chunking() -> None:
    dataset = get_dataset()
    dataset.validate(feedback=False)
    for item in dataset:
        for chunk in item.iterate_data_chunks():
            df = chunk.df
            check_column_dtypes(df, item.table_dictionary)


def test_types_applied_on_chunking_when_imported() -> None:
    dataset = get_dataset()
    dataset.validate(feedback=False)
    dataset.import_data()
    for item in dataset:
        for chunk in item.iterate_data_chunks():
            df = chunk.df
            check_column_dtypes(df, item.table_dictionary)


if __name__ == "__main__":
    pytest.main([__file__])
