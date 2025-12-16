import random
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import pytest
from pandas import DataFrame

from valediction.datasets.datasets import Dataset
from valediction.demo import DEMO_DATA, DEMO_DICTIONARY
from valediction.dictionary.model import Column, Table
from valediction.exceptions import (
    DataDictionaryError,
    DataDictionaryImportError,
)

EXPORT_DIR = Path(__file__).parent


# Helpers
@contextmanager
def cleanup(filename: Path | str):
    try:
        yield
    finally:
        file_path = filename if isinstance(filename, Path) else EXPORT_DIR / filename
        if file_path.exists():
            file_path.unlink()


@contextmanager
def cleanup_multiple(filenames: Iterable[Path | str]):
    try:
        yield
    finally:
        for filename in filenames:
            file_path = (
                filename if isinstance(filename, Path) else EXPORT_DIR / filename
            )
            if file_path.exists():
                file_path.unlink()


def get_dataset() -> Dataset:
    dataset = Dataset.create_from(DEMO_DATA)
    dataset.import_dictionary(DEMO_DICTIONARY)
    return dataset


def get_dataset_with_random_case() -> Dataset:
    dataset = get_dataset()
    dataset.import_data()

    # Randomise Data
    for item in dataset:  # tables
        coin_flip = random.randint(0, 1)
        item.name = item.name.upper() if coin_flip else item.name.lower()

        df = item.data  # columns
        for column in df.columns:
            coin_flip = random.randint(0, 1)
            df[column] = df[column].str.upper() if coin_flip else df[column].str.lower()

    # Randomise Dictionary
    dictionary = dataset.dictionary
    for table in dictionary:
        coin_flip = random.randint(0, 1)
        table.name = table.name.upper() if coin_flip else table.name.lower()

        for column in table:
            coin_flip = random.randint(0, 1)
            column.name = column.name.upper() if coin_flip else column.name.lower()

    return dataset


# Test Dataset Creation
def test_dataset_case_untouched() -> None:
    """Test that building a dataset from DataFrames leaves case intact."""
    table_name = "TestUntouchedCase"
    df = DataFrame({"A": [1, 2, 3], "b": [4, 5, 6], "C": [7, 8, 9]})
    dataset = Dataset.create_from({"TestUntouchedCase": df})

    table = dataset[0]
    original_columns = set([column for column in df.columns])
    dataset_columns = set([column for column in table.data])

    assert dataset[table_name].name == table_name
    assert original_columns == dataset_columns


def test_dataset_strips_whitespace() -> None:
    """Test that building a dataset from DataFrames strips whitespace."""
    table_name = "TestUntouchedCase "
    cols = ["A", " B ", "C"]

    df = DataFrame({cols[0]: [1, 2, 3], cols[1]: [4, 5, 6], cols[2]: [7, 8, 9]})
    dataset = Dataset.create_from({"TestUntouchedCase": df})

    table = dataset[0]
    stripped_columns = set([col.strip() for col in cols])
    dataset_columns = set([column for column in table.data])

    assert dataset[table_name].name == table_name.strip()
    assert dataset_columns == stripped_columns


def test_dataset_names_processed_on_import() -> None:
    """Test that importing a dataset strips whitespace but leaves case intact."""
    table_name = " TestNames "
    filename = f"{table_name}.csv"
    cleanup(filename)

    cols = ["A", " B ", "c"]
    df = DataFrame({cols[0]: [1, 2, 3], cols[1]: [4, 5, 6], cols[2]: [7, 8, 9]})

    with cleanup(filename):
        df.to_csv(EXPORT_DIR / filename, index=False)
        dataset = Dataset.create_from(EXPORT_DIR / filename)
        dataset.import_data()

        item = dataset[0]
        data_table_name = item.name
        data_columns = set([col for col in item.data.columns])
        assert data_table_name == table_name.strip()
        assert data_columns == set([col.strip() for col in cols])


def test_dataset_with_random_case_constructs() -> None:
    dataset = get_dataset_with_random_case()
    assert dataset


# Test Searching
def test_dataset_lowercase_search() -> None:
    dataset = get_dataset()
    for item in dataset:
        item.name = item.name.upper()

    search_lower = dataset[0].name.lower()
    item_via_get = dataset.get(search_lower)
    item_via_index = dataset[search_lower]

    assert item_via_get
    assert item_via_index


def test_dictionary_search_case() -> None:
    dataset = get_dataset()
    dictionary = dataset.dictionary

    for table in dictionary:
        table.name = table.name.upper()

    search_lower = dictionary[0].name.lower()
    item_via_get = dictionary.get_table(search_lower)
    assert item_via_get

    item_via_index = dictionary[search_lower]
    assert item_via_index


def test_dictionary_guard_against_table_collision() -> None:
    dataset = get_dataset()
    dictionary = dataset.dictionary

    for table in dictionary:
        table.name = table.name.upper()

    table_name_lower = dictionary[0].name.lower()
    with pytest.raises(DataDictionaryError) as e:
        dictionary.add_table(Table(table_name_lower))

    assert "already exists" in e.value.args[0]


# Test Collision
def test_dictionary_guard_against_column_collision() -> None:
    dataset = get_dataset()
    dictionary = dataset.dictionary

    for table in dictionary:
        for column in table:
            column.name = column.name.upper()

    column_name_lower = dictionary[0][0].name.lower()
    with pytest.raises(DataDictionaryError) as e:
        dictionary[0].add_column(Column(column_name_lower, order=1, data_type="int"))

    assert "already exists" in e.value.args[0]


def test_dictionary_guard_against_column_collision_whitespace() -> None:
    dataset = get_dataset()
    dictionary = dataset.dictionary

    for table in dictionary:
        for column in table:
            column.name = column.name.upper()

    column_name_lower = f" {dictionary[0][0].name.lower()} "
    with pytest.raises(DataDictionaryError) as e:
        dictionary[0].add_column(Column(column_name_lower, order=1, data_type="int"))

    assert "already exists" in e.value.args[0]


# Test Dictionary adds as is
def test_dictionary_adds_table_as_is() -> None:
    dataset = get_dataset()
    dictionary = dataset.dictionary

    table_name = "NewTable "
    column_name = "NewColumn "

    dictionary.add_table(
        Table(
            table_name,
            columns=[Column(column_name, order=1, data_type="int", primary_key=1)],
        )
    )

    table = dictionary[table_name.strip()]
    assert table.name == table_name.strip()


def test_dictionary_adds_column_as_is() -> None:
    dataset = get_dataset()
    dictionary = dataset.dictionary

    table_name = "NewTable "
    column_name = "NewColumn "

    dictionary.add_table(
        Table(
            table_name,
            columns=[Column(column_name, order=1, data_type="int", primary_key=1)],
        )
    )

    table = dictionary[table_name]
    column = table.get_column(column_name.strip())
    assert column.name == column_name.strip()


# Test validation case-insensitivity
def test_lowercase_dictionary_tables() -> None:
    dataset = get_dataset()
    dictionary = dataset.dictionary

    for table in dictionary:
        table.name = table.name.lower()

    dataset.validate(feedback=False)


def test_lowercase_dictionary_columns() -> None:
    dataset = get_dataset()
    dictionary = dataset.dictionary

    for table in dictionary:
        for column in table:
            column.name = column.name.lower()

    dataset.validate(feedback=False)


def test_lowercase_data_tables() -> None:
    dataset = get_dataset()
    dataset.import_data()

    for item in dataset:
        item.name = item.name.lower()

    dataset.validate(feedback=False)


def test_lowercase_data_columns() -> None:
    dataset = get_dataset()
    dataset.import_data()

    for item in dataset:
        for column in item.table_dictionary:
            column.name = column.name.lower()

    dataset.validate(feedback=False)


# Test dictionary getters
def test_get_table() -> None:
    dataset = get_dataset()
    dictionary = dataset.dictionary
    for table in dictionary:
        table.name = table.name.upper()
        assert dictionary.get_table(table.name.lower())


def test_get_columns() -> None:
    dataset = get_dataset()
    dictionary = dataset.dictionary
    for table in dictionary:
        table.name = table.name.upper()
        for column in table:
            column.name = column.name.upper()
            assert dictionary.get_table(table.name.lower()).get_column(
                column.name.lower()
            )


def test_dictionary_get_table_strips_whitespace_and_ignores_case() -> None:
    dataset = get_dataset()
    d = dataset.dictionary

    name = d[0].name
    assert d.get_table(f"  {name.swapcase()}  ")
    assert d[f"  {name.swapcase()}  "]


def test_table_get_column_strips_whitespace_and_ignores_case() -> None:
    dataset = get_dataset()
    table = dataset.dictionary[0]

    col_name = table[0].name
    assert table.get_column(f"  {col_name.swapcase()}  ").name == col_name
    assert table[f"  {col_name.swapcase()}  "].name == col_name


def test_table_remove_column_ignores_case() -> None:
    dataset = get_dataset()
    table = dataset.dictionary[0]

    col_name = table[0].name
    table.remove_column(col_name.swapcase())
    assert table.index_of(col_name) is None


# Test Dictionary Collisions
def test_dictionary_catches_table_collision() -> None:
    dataset = get_dataset()
    dictionary = dataset.dictionary

    for table in dictionary:
        table.name = table.name.upper()

    table_name_lower = dictionary[0].name.lower()
    with pytest.raises(DataDictionaryError) as e:
        dictionary.add_table(Table(table_name_lower))

    assert "already exists" in e.value.args[0]


def test_dictionary_catches_column_collision() -> None:
    dataset = get_dataset()
    dictionary = dataset.dictionary

    for table in dictionary:
        for column in table:
            column.name = column.name.upper()

    final_column = dictionary[0][-1]

    with pytest.raises(DataDictionaryError) as e:
        dictionary[0].add_column(
            Column(
                name=final_column.name.lower(),
                order=final_column.order + 1,
                data_type=final_column.data_type,
            )
        )

    assert "already exists" in e.value.args[0]


def test_getters_work_after_random_case_mutation() -> None:
    dataset = get_dataset_with_random_case()
    dictionary = dataset.dictionary

    # pick a table and column and try funky lookups
    table = dictionary[0]
    column = table[0]

    assert dictionary.get_table(table.name.swapcase())
    assert dictionary[table.name.swapcase()]
    assert dictionary.get_table(f"  {table.name.swapcase()}  ")

    assert table.get_column(column.name.swapcase())
    assert table[column.name.swapcase()]
    assert table.get_column(f"  {column.name.swapcase()}  ")


# Complete Test
def randomise_case(name: str) -> str:
    chars = []
    for char in name:
        coin_flip = random.randint(0, 1)
        chars.append(char.upper() if coin_flip else char.lower())
    return "".join(chars)


def test_all_high_level_functions_for_case_insensitivity():
    # Set filenames
    export_dictionary_filename = (
        "test_all_high_level_functions_for_case_insensitivity_DICTIONARY"
    )

    # Import
    dataset = get_dataset()
    dataset.import_data()

    dictionary = dataset.dictionary
    dictionary.name = export_dictionary_filename

    # Randomise Case
    for item in dataset:
        item.name = randomise_case(item.name)

        # column rename dictionary
        col_rename_dict = {
            column: randomise_case(column) for column in item.data.columns
        }
        item.data.rename(columns=col_rename_dict, inplace=True)

    # Test Validation
    dataset.validate(feedback=False)
    dataset.check()

    # Export Data
    table_exports = [EXPORT_DIR / f"{table.name}.csv" for table in dictionary]
    with cleanup_multiple(table_exports):
        dataset.import_data()
        dataset.validate(feedback=False)
        dataset.check()
        dataset.export_data(directory=EXPORT_DIR, overwrite=True)

    # Test Dictionary Export & Reimport
    with cleanup(f"{export_dictionary_filename}.xlsx"):
        dictionary.export_dictionary(
            directory=EXPORT_DIR, filename=export_dictionary_filename, overwrite=True
        )
        dataset = get_dataset()
        dataset.import_dictionary(
            dictionary=EXPORT_DIR / f"{export_dictionary_filename}.xlsx"
        )
        dataset.validate(feedback=False)
        dataset.check()

    # # Test Dictionary Generation & Export
    with cleanup(f"{export_dictionary_filename}.xlsx"):
        dataset.generate_dictionary(
            dictionary_name=export_dictionary_filename, feedback=False
        )

        with pytest.raises(DataDictionaryError) as e:
            dataset.validate()
            assert "no Primary Key column" in e

        dataset.export_dictionary(
            directory=EXPORT_DIR, filename=export_dictionary_filename, overwrite=True
        )
        with pytest.raises(DataDictionaryImportError) as e:
            dataset.import_dictionary(EXPORT_DIR / f"{export_dictionary_filename}.xlsx")
            assert "no Primary Key column" in e


if __name__ == "__main__":
    pytest.main([__file__])
