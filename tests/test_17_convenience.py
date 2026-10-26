from pathlib import Path

import pandas as pd
import pytest

from valediction.convenience import create_dataset, validate
from valediction.datasets.datasets import Dataset
from valediction.demo import DEMO_DATA, DEMO_DICTIONARY, demo_dictionary
from valediction.exceptions import DataDictionaryError


@pytest.mark.parametrize("data_as_string", [False, True], ids=["path", "str"])
@pytest.mark.parametrize("dictionary_type", ["object", "path", "str"])
def test_create_dataset_accepts_demo_path_and_dictionary_types(
    data_as_string: bool, dictionary_type: str
) -> None:
    data = str(DEMO_DATA) if data_as_string else DEMO_DATA
    dictionary = {
        "object": demo_dictionary(),
        "path": DEMO_DICTIONARY,
        "str": str(DEMO_DICTIONARY),
    }[dictionary_type]

    dataset = create_dataset(data=data, dictionary=dictionary)

    assert isinstance(dataset, Dataset)
    assert dataset.dictionary is not None
    assert len(dataset) == 4


def test_create_dataset_can_defer_dictionary_import() -> None:
    dataset = create_dataset(data=DEMO_DATA)

    assert dataset.dictionary is None
    assert len(dataset) == 4


def test_validate_demo_paths_with_explicit_arguments() -> None:
    dataset = validate(
        dataset=str(DEMO_DATA),
        dictionary=str(DEMO_DICTIONARY),
        import_data=False,
        chunk_size=2,
        feedback=False,
        fail_fast=False,
    )

    assert dataset.dictionary is not None
    assert len(dataset) == 4
    assert all(item.validated and item.is_path for item in dataset)


def test_validate_imports_demo_path_data_with_dictionary_object() -> None:
    dataset = validate(
        dataset=DEMO_DATA,
        dictionary=demo_dictionary(),
        import_data=True,
        chunk_size=None,
        feedback=False,
        fail_fast=True,
    )

    assert len(dataset) == 4
    assert all(item.validated and item.is_dataframe for item in dataset)


def test_validate_can_defer_dictionary_dtypes() -> None:
    dataset = validate(
        dataset=DEMO_DATA,
        dictionary=demo_dictionary(),
        import_data=True,
        apply_dtypes=False,
        feedback=False,
    )

    assert all(item.validated and item.is_dataframe for item in dataset)
    assert all(
        not item._dtypes_applied
        and all(str(dtype) == "string" for dtype in item.data.dtypes)
        for item in dataset
    )


def test_validate_prepared_dataset_can_defer_dtypes_during_import() -> None:
    prepared = create_dataset(data=DEMO_DATA, dictionary=demo_dictionary())
    validate(dataset=prepared, feedback=False)

    dataset = validate(
        dataset=prepared,
        import_data=True,
        apply_dtypes=False,
        feedback=False,
    )

    assert all(
        item.validated and all(str(dtype) == "string" for dtype in item.data.dtypes)
        for item in dataset
    )


def test_validate_prepared_dataset_uses_attached_dictionary() -> None:
    prepared = create_dataset(data=DEMO_DATA, dictionary=demo_dictionary())

    dataset = validate(dataset=prepared, feedback=False, chunk_size=2)

    assert dataset is prepared
    assert all(item.validated for item in dataset)


def test_validate_accepts_dictionary_of_demo_dataframes() -> None:
    data = {path.stem: pd.read_csv(path) for path in Path(DEMO_DATA).glob("*.csv")}

    dataset = validate(
        dataset=data,
        dictionary=DEMO_DICTIONARY,
        import_data=True,
        chunk_size=2,
        feedback=False,
    )

    assert len(dataset) == 4
    assert all(item.validated and item.is_dataframe for item in dataset)


def test_validate_requires_dictionary_for_unprepared_dataset() -> None:
    with pytest.raises(
        DataDictionaryError, match="dictionary argument must be provided"
    ):
        validate(dataset=DEMO_DATA, feedback=False)


if __name__ == "__main__":
    pytest.main([__file__])
