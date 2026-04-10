import pytest
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from valediction import demo
from valediction.datasets.datasets import Dataset
from valediction.dictionary.model import Column, Dictionary, Table
from valediction.exceptions import DataIntegrityError
from valediction.integrity import Config

TABLE_NAME = "TEST"


def get_mixed_dtype_dataset() -> Dataset:
    # returns a dataframe with: int, str, date, datetime, float, bool
    df = DataFrame(
        {
            "INT": [1, 2, 3],
            "STR": ["a", "b", "c"],
            "DATE": ["2022-01-01", "2022-02-01", "2022-03-01"],
            "TIMESTAMP": [
                "2022-01-01 00:00:00",
                "2022-02-01 00:00:00",
                "2022-03-01 00:00:00",
            ],
            "FLOAT": [1.1, 2.2, 3.3],
        }
    )

    dictionary = Dictionary(
        name="Test Dataset",
        tables=[
            Table(
                name=TABLE_NAME,
                columns=[
                    Column(name="INT", order=1, data_type="int", primary_key=1),
                    Column(name="STR", order=2, data_type="str", length=1),
                    Column(name="DATE", order=3, data_type="date"),
                    Column(name="TIMESTAMP", order=4, data_type="timestamp"),
                    Column(name="FLOAT", order=5, data_type="float"),
                ],
            )
        ],
    )

    dataset = Dataset.create_from({TABLE_NAME: df})
    dataset.import_dictionary(dictionary)
    return dataset


def test_validate_with_dtypes_set() -> None:
    dataset = get_mixed_dtype_dataset()
    dataset.validate(feedback=False)
    dataset.check()


def test_validate_int_pk_catches_null() -> None:
    dataset = get_mixed_dtype_dataset()
    dataset[TABLE_NAME].data["INT"] = [1, 2, None]
    dataset.validate(feedback=False)

    with pytest.raises(DataIntegrityError):
        dataset.check()


def test_validate_int_passed_to_text() -> None:
    dataset = get_mixed_dtype_dataset()
    dataset[TABLE_NAME].data["STR"] = [1, 2, 3]
    dataset.validate(feedback=False)
    dataset.check()


def test_validate_int_passed_to_text_with_forbidden() -> None:
    dataset = get_mixed_dtype_dataset()
    dataset[TABLE_NAME].data["STR"] = [1, 2, 3]

    with Config() as config:
        config.forbidden_characters = ["3", 2]
        dataset.validate(feedback=False)

    with pytest.raises(DataIntegrityError):
        dataset.check()


def test_validate_twice_on_demo() -> None:
    dataset = Dataset.create_from(demo.DEMO_DATA)
    dataset.import_dictionary(demo.DEMO_DICTIONARY)
    dataset.import_data()

    # 1st Pass
    dataset.validate(feedback=False)
    dataset.check()
    first = {item.name: item.data.copy(deep=True) for item in dataset}

    # 2nd pass
    dataset.validate(feedback=False)
    dataset.check()
    second = {item.name: item.data.copy(deep=True) for item in dataset}

    assert first.keys() == second.keys()
    for table in first.keys():
        assert_frame_equal(
            first[table],
            second[table],
            check_dtype=True,
            check_like=False,
        )


if __name__ == "__main__":
    pytest.main([__file__])
