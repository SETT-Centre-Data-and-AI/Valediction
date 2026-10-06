from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from valediction import Dataset, demo
from valediction.datasets.datasets import DatasetItem
from valediction.dictionary.model import Column, Dictionary, Table
from valediction.integrity import Config

EXPORT_DIR = Path(__file__).parent


# Get Helpers
@contextmanager
def cleanup(items: DatasetItem | Iterable[DatasetItem]):
    items: list[DatasetItem] = [items] if isinstance(items, DatasetItem) else items

    paths = [EXPORT_DIR / f"{item.name}.csv" for item in items]

    # Pre-clean
    for p in paths:
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    try:
        yield paths[0] if len(paths) == 1 else paths
    finally:
        for p in paths:
            try:
                p.unlink()
            except FileNotFoundError:
                pass


def export_exists(item: DatasetItem):
    path = EXPORT_DIR / f"{item.name}.csv"
    return path.exists()


def get_dataset() -> Dataset:
    dataset = Dataset.create_from(demo.DEMO_DATA)
    dataset.import_dictionary(demo.DEMO_DICTIONARY)
    return dataset


# Tests at item level
def test_item_export_post_validation() -> None:
    dataset = get_dataset()
    dataset.import_data()
    dataset.validate(feedback=False)
    item = dataset[0]

    with cleanup(item):
        item.export_data(directory=EXPORT_DIR)


def test_raises_item_export_no_validation() -> None:
    dataset = get_dataset()
    dataset.import_data()
    item = dataset[0]

    with cleanup(item):
        with pytest.raises(ValueError):
            item.export_data(directory=EXPORT_DIR)


def test_raises_item_export_no_import() -> None:
    dataset = get_dataset()
    item = dataset[0]

    with cleanup(item):
        with pytest.raises(ValueError):
            item.export_data(directory=EXPORT_DIR)


def test_raises_item_export_validated_but_no_import() -> None:
    dataset = get_dataset()
    dataset.validate(feedback=False)
    item = dataset[0]

    with cleanup(item):
        with pytest.raises(ValueError):
            item.export_data(directory=EXPORT_DIR)


def test_raises_item_export_imported_but_not_validated() -> None:
    dataset = get_dataset()
    dataset.import_data()
    item = dataset[0]

    with cleanup(item):
        with pytest.raises(ValueError):
            item.export_data(directory=EXPORT_DIR)


def test_item_export_validation_override() -> None:
    dataset = get_dataset()
    dataset.import_data()
    item = dataset[0]

    with cleanup(item):
        item.export_data(directory=EXPORT_DIR, enforce_validation=False)


def test_item_export_raises_on_conflict() -> None:
    dataset = get_dataset()
    dataset.import_data()
    dataset.validate(feedback=False)
    item = dataset[0]

    with cleanup(item):
        item.export_data(directory=EXPORT_DIR)
        with pytest.raises(ValueError):
            item.export_data(directory=EXPORT_DIR, enforce_validation=False)


def test_item_export_overwrite() -> None:
    dataset = get_dataset()
    dataset.import_data()
    dataset.validate(feedback=False)
    item = dataset[0]

    with cleanup(item):
        item.export_data(directory=EXPORT_DIR)
        item.export_data(directory=EXPORT_DIR, enforce_validation=False, overwrite=True)


# Tests at dataset level
def test_dataset_export_post_validation() -> None:
    dataset = get_dataset()
    dataset.import_data()
    dataset.validate(feedback=False)

    with cleanup(dataset):
        dataset.export_data(directory=EXPORT_DIR)


def test_dataset_skips_unvalidated() -> None:
    dataset = get_dataset()
    dataset.import_data()
    indexes = [0, 1]

    validated = [dataset[i] for i in indexes]
    unvalidated = [dataset[i] for i in range(len(dataset)) if i not in indexes]

    for item in validated:
        item.validate(feedback=False)

    with cleanup(dataset):
        dataset.export_data(directory=EXPORT_DIR)
        for item in validated:
            assert export_exists(item)
        for item in unvalidated:
            assert not export_exists(item)


def test_dataset_export_skips_unimported() -> None:
    dataset = get_dataset()
    dataset.validate(feedback=False)
    indexes = [0, 1]

    imported = [dataset[i] for i in indexes]
    unimported = [dataset[i] for i in range(len(dataset)) if i not in indexes]

    for item in imported:
        item.import_data()

    with cleanup(dataset):
        dataset.export_data(directory=EXPORT_DIR)

        for item in imported:
            assert export_exists(item)
        for item in unimported:
            assert not export_exists(item)


def test_dataset_export_validation_override() -> None:
    dataset = get_dataset()
    dataset.import_data()

    with cleanup(dataset):
        dataset.export_data(directory=EXPORT_DIR, enforce_validation=False)

        for item in dataset:
            assert export_exists(item)


def test_dataset_export_raises_on_conflict() -> None:
    dataset = get_dataset()
    dataset.import_data()
    dataset.validate(feedback=False)

    with cleanup(dataset):
        dataset.export_data(directory=EXPORT_DIR)

        for item in dataset:
            assert export_exists(item)

        with pytest.raises(ValueError):
            dataset.export_data(directory=EXPORT_DIR)


def test_dataset_export_overwrite() -> None:
    dataset = get_dataset()
    dataset.import_data()
    dataset.validate(feedback=False)

    with cleanup(dataset):
        dataset.export_data(directory=EXPORT_DIR)

        for item in dataset:
            assert export_exists(item)

        dataset.export_data(directory=EXPORT_DIR, overwrite=True)


@pytest.mark.parametrize("infer_format", [False, True], ids=["explicit", "inferred"])
@pytest.mark.parametrize("writer", ["valediction", "pandas"])
@pytest.mark.parametrize("reader", ["path", "import_data", "pandas"])
@pytest.mark.parametrize("midnight", [False, True], ids=["with_time", "midnight"])
def test_timestamp_round_trip(
    tmp_path: Path, infer_format: bool, writer: str, reader: str, midnight: bool
) -> None:
    datetime_format = "%d/%m/%Y %H:%M:%S"
    values = pd.Series(
        ["22/08/2023 00:00:00", "23/08/2023 00:00:00"]
        if midnight
        else ["22/08/2023 22:13:07", "23/08/2023 09:05:11"]
    )
    expected = pd.to_datetime(values, format=datetime_format).rename("OBSERVATION_TIME")
    frame = pd.DataFrame(
        {
            "PATIENT_HASH": ["PCB1AC7AEFBC", "PCB1AC7AEFBC"],
            "OBSERVATION_TIME": values if infer_format else expected.copy(),
            "OBSERVATION_TYPE": ["Diastolic_BP", "Diastolic_BP"],
            "RESULT": [71.8, 72.1],
        }
    )
    dictionary = demo.demo_dictionary()
    dictionary.get_table("vitals").get_column("OBSERVATION_TIME").datetime_format = (
        None if infer_format else datetime_format
    )
    dataset = Dataset.create_from({"VITALS": frame})
    dataset.import_dictionary(dictionary)
    dataset.validate(feedback=False)
    assert dataset.check()
    assert_series_equal(dataset["VITALS"].data["OBSERVATION_TIME"], expected)
    assert (
        dictionary.get_table("vitals").get_column("OBSERVATION_TIME").datetime_format
        == datetime_format
    )

    csv_path = tmp_path / "VITALS.csv"
    if writer == "valediction":
        dataset.export_data(directory=tmp_path)
    else:
        dataset["VITALS"].data.to_csv(csv_path, index=False)
    if writer == "valediction":
        expected_text = expected.map(lambda value: value.isoformat()).tolist()
    else:
        csv_format = "%Y-%m-%d" if midnight else "%Y-%m-%d %H:%M:%S"
        expected_text = expected.dt.strftime(csv_format).tolist()
    assert pd.read_csv(csv_path)["OBSERVATION_TIME"].tolist() == expected_text

    restored = Dataset.create_from(
        {"VITALS": pd.read_csv(csv_path)} if reader == "pandas" else csv_path
    )
    restored.import_dictionary(deepcopy(dictionary))
    if reader == "import_data":
        restored.import_data()
    restored.validate(feedback=False, chunk_size=1)
    assert restored.check()
    assert restored["VITALS"].validated
    assert not restored["VITALS"].issues
    if reader == "path":
        restored.import_data()
    assert_series_equal(restored["VITALS"].data["OBSERVATION_TIME"], expected)
    assert (
        restored["VITALS"]
        .table_dictionary.get_column("OBSERVATION_TIME")
        .datetime_format
        == datetime_format
    )


@pytest.mark.parametrize("reader", ["path", "import_data", "chunks"])
@pytest.mark.parametrize("declared", [False, True])
@pytest.mark.parametrize("month_first", [False, True])
def test_date_promotion_csv_round_trip(
    tmp_path: Path, reader: str, declared: bool, month_first: bool
) -> None:
    selected_format = (
        ("%m/%d/%Y %H:%M:%S" if month_first else "%d/%m/%Y %H:%M:%S")
        if declared
        else None
    )
    values = [
        "2023-08-22",
        "2023/08/22",
        "08/22/2023" if month_first else "22/08/2023",
        "08-22-2023" if month_first else "22-08-2023",
        "01/02/2023",
        "2023-08-23T09:05:11.123456789",
        None,
    ]
    expected = pd.Series(
        [
            *[pd.Timestamp("2023-08-22")] * 4,
            pd.Timestamp("2023-01-02" if month_first else "2023-02-01"),
            pd.Timestamp("2023-08-23 09:05:11.123456789"),
            pd.NaT,
        ],
        name="TIMESTAMP",
    )
    dictionary = Dictionary(
        name="Dates",
        tables=[
            Table(
                name="DATES",
                columns=[
                    Column(name="ID", order=1, data_type="int", primary_key=1),
                    Column(
                        name="TIMESTAMP",
                        order=2,
                        data_type="timestamp",
                        datetime_format=selected_format,
                    ),
                ],
            )
        ],
    )
    csv_path = tmp_path / "DATES.csv"
    pd.DataFrame({"ID": range(1, len(values) + 1), "TIMESTAMP": values}).to_csv(
        csv_path, index=False
    )
    restored = Dataset.create_from(csv_path)
    restored.import_dictionary(dictionary)
    if reader == "import_data":
        restored.import_data()
    restored.validate(feedback=False, chunk_size=1)
    assert restored.check()
    if reader == "chunks":
        actual = pd.concat(
            [chunk.df for chunk in restored["DATES"].iterate_data_chunks(1)],
            ignore_index=True,
        )["TIMESTAMP"]
    else:
        restored.import_data()
        actual = restored["DATES"].data["TIMESTAMP"]
    assert_series_equal(actual, expected)
    restored.import_data()
    output = tmp_path / "output"
    restored.export_data(output)
    raw = pd.read_csv(output / "DATES.csv", dtype="string", keep_default_na=False)
    assert raw["TIMESTAMP"].tolist() == [
        value.isoformat() if pd.notna(value) else "" for value in expected
    ]
    reimported = Dataset.create_from(output / "DATES.csv")
    reimported.import_dictionary(deepcopy(dictionary))
    reimported.validate(feedback=False, chunk_size=1)
    assert reimported.check()
    reimported.import_data()
    assert_series_equal(reimported["DATES"].data["TIMESTAMP"], expected)


@pytest.mark.parametrize("reader", ["path", "import_data", "chunks"])
@pytest.mark.parametrize("timezone", [None, "UTC", "UTC+05:30"])
@pytest.mark.parametrize("selected_format", [None, "%d/%m/%Y %H:%M:%S"])
def test_datetime_fidelity_round_trip(
    tmp_path: Path, reader: str, timezone: str | None, selected_format: str | None
) -> None:
    timestamp = pd.Series(
        [
            pd.Timestamp("2023-08-22 00:00:00"),
            pd.Timestamp("2023-08-23 09:05:11.123456789"),
            pd.Timestamp("2023-08-24 09:05:11.123456"),
            pd.NaT,
        ],
        name="TIMESTAMP",
    )
    if timezone:
        timestamp = timestamp.dt.tz_localize(timezone)
    frame = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "TIMESTAMP": timestamp,
            "DATE": pd.to_datetime(["2023-08-22", "2023-08-23", None, "2023-08-24"]),
            "TEXT": ['a,"quoted"', "line\nbreak", "NA", None],
            "FLOAT": [0.30000000000000004, None, 0.12345678912345678, -2.5],
        }
    )
    dictionary = Dictionary(
        name="Fidelity",
        tables=[
            Table(
                name="FIDELITY",
                columns=[
                    Column(name="ID", order=1, data_type="int", primary_key=1),
                    Column(
                        name="TIMESTAMP",
                        order=2,
                        data_type="timestamp",
                        datetime_format=selected_format,
                    ),
                    Column(
                        name="DATE",
                        order=3,
                        data_type="date",
                        datetime_format="%d/%m/%Y",
                    ),
                    Column(name="TEXT", order=4, data_type="text", length=100),
                    Column(name="FLOAT", order=5, data_type="float"),
                ],
            )
        ],
    )
    dataset = Dataset.create_from({"FIDELITY": frame})
    dataset.import_dictionary(dictionary)
    dataset.validate(feedback=False)
    assert dataset.check()
    expected = dataset["FIDELITY"].data.copy(deep=True)
    formats_before = [
        column.datetime_format for column in dictionary.get_table("FIDELITY")
    ]
    dataset.export_data(tmp_path)
    assert_frame_equal(dataset["FIDELITY"].data, expected, check_exact=True)
    assert [
        column.datetime_format for column in dictionary.get_table("FIDELITY")
    ] == formats_before
    csv_path = tmp_path / "FIDELITY.csv"
    raw = pd.read_csv(csv_path, dtype="string", keep_default_na=False)
    assert raw["TIMESTAMP"].tolist() == [
        value.isoformat() if pd.notna(value) else "" for value in timestamp
    ]
    assert raw["DATE"].tolist() == ["22/08/2023", "23/08/2023", "", "24/08/2023"]
    restored = Dataset.create_from(csv_path)
    restored.import_dictionary(deepcopy(dictionary))
    if reader == "import_data":
        restored.import_data()
    restored.validate(feedback=False, chunk_size=1)
    assert restored.check()
    if reader == "chunks":
        actual = pd.concat(
            [chunk.df for chunk in restored["FIDELITY"].iterate_data_chunks(1)],
            ignore_index=True,
        )
    else:
        restored.import_data()
        actual = restored["FIDELITY"].data
    assert_frame_equal(actual, expected, check_exact=True)
    assert actual["FLOAT"].dropna().tolist() == expected["FLOAT"].dropna().tolist()


@pytest.mark.parametrize("reader", ["path", "import_data", "chunks"])
def test_mixed_offset_csv_fidelity(tmp_path: Path, reader: str) -> None:
    timestamp = pd.Series(
        [
            pd.Timestamp("2023-01-22T00:00:00.123456789+00:00"),
            pd.Timestamp("2023-08-23T09:05:11.987654321+01:00"),
            pd.NaT,
        ],
        dtype=object,
        name="TIMESTAMP",
    )
    dictionary = Dictionary(
        name="Offsets",
        tables=[
            Table(
                name="OFFSETS",
                columns=[
                    Column(name="ID", order=1, data_type="int", primary_key=1),
                    Column(
                        name="TIMESTAMP",
                        order=2,
                        data_type="timestamp",
                        datetime_format="%Y-%m-%dT%H:%M:%S.%f%z",
                    ),
                ],
            )
        ],
    )
    dataset = Dataset.create_from(
        {"OFFSETS": pd.DataFrame({"ID": [1, 2, 3], "TIMESTAMP": timestamp})}
    )
    dataset.import_dictionary(dictionary)
    dataset.validate(feedback=False)
    assert dataset.check()
    dataset.export_data(tmp_path)
    raw = pd.read_csv(tmp_path / "OFFSETS.csv", dtype="string", keep_default_na=False)
    assert raw["TIMESTAMP"].tolist() == [
        value.isoformat() if pd.notna(value) else "" for value in timestamp
    ]
    restored = Dataset.create_from(tmp_path / "OFFSETS.csv")
    restored.import_dictionary(deepcopy(dictionary))
    if reader == "import_data":
        restored.import_data()
    restored.validate(feedback=False, chunk_size=1)
    assert restored.check()
    if reader == "chunks":
        actual = pd.concat(
            [chunk.df for chunk in restored["OFFSETS"].iterate_data_chunks(1)],
            ignore_index=True,
        )["TIMESTAMP"]
    else:
        restored.import_data()
        actual = restored["OFFSETS"].data["TIMESTAMP"]
    assert_series_equal(actual, timestamp)


@pytest.mark.parametrize("reader", ["path", "import_data", "chunks", "pandas"])
@pytest.mark.parametrize("source", ["native", "text"])
def test_integer_csv_fidelity(tmp_path: Path, reader: str, source: str) -> None:
    expected = pd.DataFrame(
        {
            "ID": pd.Series(range(1, 9), dtype="Int64"),
            "VALUE": pd.Series(
                [
                    9223372036854775807,
                    -9223372036854775808,
                    9007199254740993,
                    None,
                    1,
                    100,
                    None,
                    None,
                ],
                dtype="Int64",
            ),
            "TEXT": pd.Series(
                ["NA", None, "literal", None, "kept", "kept", None, None],
                dtype="string",
            ),
        }
    )
    frame = expected.copy(deep=True)
    if source == "text":
        frame["VALUE"] = [
            "9223372036854775807.0",
            "-9.223372036854775808e18",
            "9007199254740993",
            "",
            "1.0",
            "1e2",
            "null",
            "missing",
        ]
        frame["TEXT"] = ["NA", "none", "literal", "", "kept", "kept", "null", "missing"]
    dictionary = Dictionary(
        name="Numbers",
        tables=[
            Table(
                name="NUMBERS",
                columns=[
                    Column(name="ID", order=1, data_type="int", primary_key=1),
                    Column(name="VALUE", order=2, data_type="int"),
                    Column(name="TEXT", order=3, data_type="text", length=30),
                ],
            )
        ],
    )
    csv_path = tmp_path / "NUMBERS.csv"
    frame.to_csv(csv_path, index=False)
    with Config() as config:
        config.null_values = ["", "null", "none", "missing"]
        restored = Dataset.create_from(
            {"NUMBERS": pd.read_csv(csv_path, dtype="string", keep_default_na=False)}
            if reader == "pandas"
            else csv_path
        )
        restored.import_dictionary(dictionary)
        if reader == "import_data":
            restored.import_data()
        restored.validate(feedback=False, chunk_size=1)
        assert restored.check()
        if reader == "chunks":
            actual = pd.concat(
                [chunk.df for chunk in restored["NUMBERS"].iterate_data_chunks(1)],
                ignore_index=True,
            )
        else:
            restored.import_data()
            actual = restored["NUMBERS"].data
        assert_frame_equal(actual, expected, check_exact=True)
        restored.import_data()
        output = tmp_path / "output"
        restored.export_data(output)
        raw = pd.read_csv(output / "NUMBERS.csv", dtype="string", keep_default_na=False)
        assert raw["VALUE"].tolist() == [
            "9223372036854775807",
            "-9223372036854775808",
            "9007199254740993",
            "",
            "1",
            "100",
            "",
            "",
        ]
        final = Dataset.create_from(output / "NUMBERS.csv")
        final.import_dictionary(deepcopy(dictionary))
        final.validate(feedback=False, chunk_size=1)
        assert final.check()
        final.import_data()
        assert_frame_equal(final["NUMBERS"].data, expected, check_exact=True)


if __name__ == "__main__":
    pytest.main([__file__])
