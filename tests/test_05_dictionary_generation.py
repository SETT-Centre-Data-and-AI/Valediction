from pathlib import Path

import pytest
from pandas import DataFrame

from valediction.data_types.data_types import DataType
from valediction.datasets.datasets import Dataset
from valediction.demo import DEMO_DATA, demo_dictionary
from valediction.exceptions import DuplicateHeaderError
from valediction.integrity import Config

DICT_NAME = "TEST"


# Parameters
@pytest.fixture(params=[True, False], ids=["feedback", "no_feedback"])
def feedback(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(params=[True, False], ids=["debug", "no_debug"])
def debug(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(
    params=[None, 100, 1_000, 1_000_000],
    ids=["no_chunk", "chunk_100", "chunk_1k", "chunk_1m"],
)
def chunk_size(request: pytest.FixtureRequest) -> int | None:
    return request.param


# Helpers
def create_dataset() -> Dataset:
    dataset = Dataset.create_from(DEMO_DATA)
    return dataset


def test_generate_dictionary(feedback) -> None:
    dataset = create_dataset()
    dictionary = dataset.generate_dictionary(
        dictionary_name=DICT_NAME, feedback=feedback
    )

    assert dictionary
    assert len(dictionary) == 4


def test_generate_dictionary_with_mixed_timezone_timestamps(debug) -> None:
    df = DataFrame(
        {
            "ID": ["1", "2", "3"],
            "OBS_TIMESTAMP": [
                "2021-11-27T16:56:00+00:00",
                "2021-06-27T17:56:00+01:00",
                "2021-11-28T08:15:00+00:00",
            ],
        }
    )
    dataset = Dataset.create_from({"OBSERVATIONS": df})

    dictionary = dataset.generate_dictionary(
        dictionary_name=DICT_NAME,
        feedback=False,
        debug=debug,
    )
    timestamp_column = dictionary.get_table("OBSERVATIONS").get_column("OBS_TIMESTAMP")

    assert timestamp_column.data_type is DataType.TIMESTAMP
    assert timestamp_column.datetime_format == "%Y-%m-%dT%H:%M:%S%z"


@pytest.mark.parametrize("size", [None, 1, 2])
@pytest.mark.parametrize("conflict", [False, True])
def test_generated_calendar_consistency(
    tmp_path: Path, size: int | None, conflict: bool
) -> None:
    values = ["01/02/2023", "08/23/2023", "22/08/2023" if conflict else "2023-08-24"]
    path = tmp_path / "CALENDAR.csv"
    DataFrame({"ID": [1, 2, 3], "VALUE": values}).to_csv(path, index=False)
    dataset = Dataset.create_from(path)
    dictionary = dataset.generate_dictionary(
        feedback=False, chunk_size=size, primary_keys={"CALENDAR": ["ID"]}
    )
    column = dictionary.get_table("CALENDAR").get_column("VALUE")
    assert column.data_type is (DataType.TEXT if conflict else DataType.DATE)
    assert column.datetime_format == (None if conflict else "%m/%d/%Y")
    if not conflict:
        dataset.validate(feedback=False, chunk_size=size)
        assert dataset.check()
        dataset.import_data()
        assert str(dataset["CALENDAR"].data["VALUE"].iloc[0]) == "2023-01-02 00:00:00"


@pytest.mark.parametrize("month_first", [False, True])
@pytest.mark.parametrize("path_input", [False, True])
def test_calendar_configured_tie_break_across_workflows(
    tmp_path: Path, month_first: bool, path_input: bool
) -> None:
    frame = DataFrame({"ID": [1, 2], "VALUE": ["01/02/2023", "02/01/2023"]})
    path = tmp_path / "CALENDAR.csv"
    frame.to_csv(path, index=False)
    with Config() as config:
        formats = ["%m/%d/%Y", "%d/%m/%Y"] if month_first else ["%d/%m/%Y", "%m/%d/%Y"]
        config.date_formats = {fmt: DataType.DATE for fmt in formats}
        dataset = Dataset.create_from(path if path_input else {"CALENDAR": frame})
        dictionary = dataset.generate_dictionary(
            feedback=False, chunk_size=1, primary_keys={"CALENDAR": ["ID"]}
        )
        column = dictionary.get_table("CALENDAR").get_column("VALUE")
        assert column.datetime_format == formats[0]
        column.datetime_format = None
        dataset.validate(feedback=False, chunk_size=1)
        assert dataset.check()
        assert column.datetime_format == formats[0]
        dataset.apply_dictionary()
        assert str(dataset["CALENDAR"].data["VALUE"].iloc[0]) == (
            "2023-01-02 00:00:00" if month_first else "2023-02-01 00:00:00"
        )


def test_generated_calendar_sample_limits_evidence(tmp_path: Path) -> None:
    path = tmp_path / "CALENDAR.csv"
    DataFrame({"ID": [1, 2], "VALUE": ["22/08/2023", "08/23/2023"]}).to_csv(
        path, index=False
    )
    dataset = Dataset.create_from(path)
    dictionary = dataset.generate_dictionary(
        feedback=False, sample_rows=1, primary_keys={"CALENDAR": ["ID"]}
    )
    assert (
        dictionary.get_table("CALENDAR").get_column("VALUE").data_type is DataType.DATE
    )
    dataset.validate(feedback=False, chunk_size=1)
    assert dataset.issues[0].type.value == "InconsistentDate"


def test_generate_dictionary_respects_bigint_limit() -> None:
    df = DataFrame({"BIGS": [2147483648, 2147483649]})
    dataset = Dataset.create_from({"BIG_TABLE": df})

    with Config() as config:
        config.allow_bigint = False
        dictionary = dataset.generate_dictionary(
            dictionary_name=DICT_NAME,
            feedback=False,
        )

    column = dictionary.get_table("BIG_TABLE").get_column("BIGS")

    assert column.data_type is DataType.TEXT
    assert column.length == len("2147483649")


def test_generate_dictionary_rejects_duplicate_dataframe_headers() -> None:
    df = DataFrame([["1", "2"]], columns=["ID", "ID"])
    dataset = Dataset.create_from({"DUPLICATES": df})

    with pytest.raises(DuplicateHeaderError) as exc_info:
        dataset.generate_dictionary(feedback=False)

    assert str(exc_info.value) == (
        "Duplicate column headers found in table 'DUPLICATES': \n - ID"
    )


@pytest.mark.parametrize("sample_rows", [None, 1], ids=["chunked", "sampled"])
def test_generate_dictionary_rejects_duplicate_csv_headers(
    tmp_path, sample_rows
) -> None:
    csv_path = tmp_path / "duplicates.csv"
    csv_path.write_text("ID,VALUE,ID\n1,test,2\n", encoding="utf-8")
    dataset = Dataset.create_from(csv_path)

    with pytest.raises(DuplicateHeaderError) as exc_info:
        dataset.generate_dictionary(feedback=False, sample_rows=sample_rows)

    assert str(exc_info.value) == (
        "Duplicate column headers found in table 'duplicates': \n - ID"
    )


def test_generated_dictionary_integrity(feedback) -> None:
    dataset = Dataset.create_from(DEMO_DATA)
    demo_dict = demo_dictionary()
    gen_dict = dataset.generate_dictionary("Synthetic Data")

    # Check tables
    assert len(gen_dict) == len(demo_dict)
    assert set([table.name for table in gen_dict]) == set(
        table.name for table in gen_dict
    )

    # Check Columns
    for gen_table in gen_dict:
        demo_table = demo_dict.get_table(table=gen_table.name)
        assert len(gen_table) == len(demo_table)
        assert set([column.name for column in gen_table]) == set(
            column.name for column in demo_table
        )

        # Check Data Types
        for column in gen_table:
            assert (
                column.data_type == demo_table.get_column(column=column.name).data_type
            )


def test_runtimes_always_saved(feedback) -> None:
    dataset = create_dataset()
    _dictionary = dataset.generate_dictionary(
        dictionary_name=DICT_NAME, feedback=feedback
    )

    assert (item._dictionary_runtimes for item in dataset)


def test_with_sample_rows(feedback) -> None:
    dataset = create_dataset()
    _dictionary = dataset.generate_dictionary(
        dictionary_name=DICT_NAME, feedback=feedback, sample_rows=100
    )

    assert (item._dictionary_runtimes for item in dataset)


def test_with_chunk_size(feedback, chunk_size) -> None:
    dataset = create_dataset()
    _dictionary = dataset.generate_dictionary(
        dictionary_name=DICT_NAME, feedback=feedback, chunk_size=chunk_size
    )

    assert (item._dictionary_runtimes for item in dataset)


def test_with_sample_rows_and_chunk_size(feedback, chunk_size) -> None:
    dataset = create_dataset()
    _dictionary = dataset.generate_dictionary(
        dictionary_name=DICT_NAME,
        feedback=feedback,
        chunk_size=chunk_size,
        sample_rows=100,
    )

    assert (item._dictionary_runtimes for item in dataset)


def test_with_debug(feedback, chunk_size, debug) -> None:
    dataset = create_dataset()
    _dictionary = dataset.generate_dictionary(
        dictionary_name=DICT_NAME,
        feedback=feedback,
        chunk_size=chunk_size,
        sample_rows=100,
        debug=debug,
    )

    assert (item._dictionary_runtimes for item in dataset)


if __name__ == "__main__":
    pytest.main([__file__])
