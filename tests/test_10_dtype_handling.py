import pytest
from pandas import DataFrame, Series, Timestamp, date_range, to_timedelta
from pandas.testing import assert_frame_equal

from valediction import demo
from valediction.datasets.datasets import Dataset
from valediction.dictionary.model import Column, Dictionary, Table
from valediction.exceptions import DataIntegrityError
from valediction.integrity import Config
from valediction.validation.helpers import (
    apply_data_types,
    invalid_mask_date,
    invalid_mask_datetime,
    invalid_mask_integer,
    invalid_mask_integer_out_of_range,
)
from valediction.validation.issues import IssueType, Range

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


def test_date_mask_accepts_date_and_midnight_timestamp_strings() -> None:
    values = Series(
        ["2022-01-01", "2022-02-01 00:00:00", "2022-03-01T00:00:00.000000"],
        dtype="string",
    )

    invalid = invalid_mask_date(values, fmt=None)

    assert invalid.tolist() == [False, False, False]


def test_date_mask_rejects_non_midnight_for_string_and_native_datetime() -> None:
    string_values = Series(
        ["2022-01-01 00:00:00", "2022-02-01 00:00:01"],
        dtype="string",
    )
    datetime_values = Series(
        date_range("2022-01-01", periods=2, freq="D") + to_timedelta(["0s", "1s"])
    )

    assert invalid_mask_date(string_values, fmt=None).tolist() == [False, True]
    assert invalid_mask_date(datetime_values, fmt=None).tolist() == [False, True]


def test_date_mask_rejects_timezone_aware_datetime() -> None:
    values = Series(date_range("2022-01-01", periods=2, freq="D", tz="UTC"))
    timezone_strings = Series(["2022-01-01T00:00:00+0000"], dtype="string")

    invalid = invalid_mask_date(values, fmt=None)

    assert invalid.tolist() == [True, True]
    assert invalid_mask_date(timezone_strings, fmt=None).tolist() == [True]


def test_datetime_format_is_a_hint_with_configured_format_fallback() -> None:
    date_values = Series(["2022-01-01", "2022-02-01 00:00:00"], dtype="string")
    timestamp_values = Series(
        ["2022-01-01T00:00:00.123456789", "2022-02-01 00:00:00"],
        dtype="string",
    )

    assert invalid_mask_date(date_values, fmt="%Y/%m/%d").tolist() == [False, False]
    assert invalid_mask_datetime(
        timestamp_values, fmt="%Y-%m-%d %H:%M:%S"
    ).tolist() == [False, False]


def test_timestamp_mask_accepts_midnight_and_non_midnight_native_values() -> None:
    values = Series(
        ["2022-01-01T00:00:00.123456789", "2022-01-02 00:00:00"],
        dtype="string",
    )

    invalid = invalid_mask_datetime(values, fmt=None)

    assert invalid.tolist() == [False, False]


def test_integer_mask_rejects_precision_loss_non_finite_and_int64_overflow() -> None:
    values = Series(
        [
            "9007199254740993.5",
            "inf",
            "-inf",
            "9223372036854775808",
            "-9223372036854775809",
            "9007199254740993",
            "9223372036854775807",
            "-9223372036854775808",
            "1e2",
        ],
        dtype="string",
    )

    assert invalid_mask_integer(values).tolist() == [
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]


def test_integer_mask_rejects_non_finite_native_floats() -> None:
    values = Series([1.0, float("inf"), float("-inf")])

    assert invalid_mask_integer(values).tolist() == [False, True, True]


def test_int4_range_mask_handles_decimal_and_scientific_notation() -> None:
    values = Series(
        [
            "2147483647.0",
            "-2147483648.0",
            "2147483648.0",
            "-2147483649.0",
            "1e10",
            "-1e10",
        ],
        dtype="string",
    )
    invalid_integer = invalid_mask_integer(values)

    assert invalid_integer.tolist() == [False] * 6
    assert invalid_mask_integer_out_of_range(
        values, invalid_integer_mask=invalid_integer
    ).tolist() == [False, False, True, True, True, True]


def test_integer_mask_tolerance_does_not_override_int64_bounds() -> None:
    values = Series(["9.223372036854776e18", "0.25"], dtype="string")

    assert invalid_mask_integer(values, tolerance=1e30).tolist() == [True, False]


def test_validate_rejects_high_precision_fraction_before_integer_conversion() -> None:
    dataset = get_mixed_dtype_dataset()
    dataset[TABLE_NAME].data["INT"] = Series(
        ["9007199254740993.5", "2", "3"], dtype="string"
    )

    dataset[TABLE_NAME].validate(feedback=False)

    item = dataset[TABLE_NAME]
    assert not item.validated
    issue = next(
        issue
        for issue in item.issues
        if issue.type == IssueType.TYPE_MISMATCH and issue.column == "INT"
    )
    assert issue.ranges == [Range(0, 0)]
    assert item.data["INT"].iloc[0] == "9007199254740993.5"


def test_validate_preserves_large_integer_with_mixed_numeric_text_formats() -> None:
    dataset = get_mixed_dtype_dataset()
    dataset[TABLE_NAME].data["INT"] = Series(
        ["9007199254740993.0", "2.0", "3e0"], dtype="string"
    )

    dataset[TABLE_NAME].validate(feedback=False)

    assert dataset[TABLE_NAME].data["INT"].tolist() == [9007199254740993, 2, 3]


def test_path_import_normalizes_null_tokens_before_type_conversion(tmp_path) -> None:
    csv_path = tmp_path / "NULLABLE.csv"
    DataFrame(
        {
            "ID": [1, 2, 3],
            "VALUE": ["null", " NONE ", "2"],
        }
    ).to_csv(csv_path, index=False)
    dataset = Dataset.create_from(csv_path)
    dataset.import_dictionary(
        Dictionary(
            name="Nullable integer import",
            tables=[
                Table(
                    name="NULLABLE",
                    columns=[
                        Column("ID", 1, "int", primary_key=1),
                        Column("VALUE", 2, "int"),
                    ],
                )
            ],
        )
    )

    dataset.validate(feedback=False)
    assert dataset["NULLABLE"].is_path

    dataset.import_data()

    values = dataset["NULLABLE"].data["VALUE"]
    assert values.isna().tolist() == [True, True, False]
    assert values.iloc[2] == 2


def test_apply_data_types_does_not_repeat_null_normalization() -> None:
    dataset = get_mixed_dtype_dataset()
    converted = apply_data_types(
        DataFrame({"STR": Series(["null", "x", "y"], dtype="string")}),
        dataset[TABLE_NAME].table_dictionary,
    )

    assert converted["STR"].iloc[0] == "null"


def test_dataframe_dictionary_source_normalizes_null_tokens() -> None:
    dataset = Dataset.create_from(
        {
            TABLE_NAME: DataFrame(
                {
                    "ID": [1, 2, 3],
                    "VALUE": ["null", " NONE ", "2"],
                }
            )
        }
    )
    dataset.import_dictionary(
        Dictionary(
            name="Nullable dataframe source",
            tables=[
                Table(
                    name=TABLE_NAME,
                    columns=[
                        Column("ID", 1, "int", primary_key=1),
                        Column("VALUE", 2, "int"),
                    ],
                )
            ],
        )
    )

    dataset.validate(feedback=False)

    item = dataset[TABLE_NAME]
    assert item.validated
    assert item.data["VALUE"].isna().tolist() == [True, True, False]
    assert item.data["VALUE"].iloc[2] == 2


def test_path_chunk_iteration_normalizes_null_tokens_before_type_conversion(
    tmp_path,
) -> None:
    csv_path = tmp_path / "NULLABLE.csv"
    DataFrame(
        {
            "ID": [1, 2, 3],
            "VALUE": ["null", " NONE ", "2"],
        }
    ).to_csv(csv_path, index=False)
    dataset = Dataset.create_from(csv_path)
    dataset.import_dictionary(
        Dictionary(
            name="Nullable integer iteration",
            tables=[
                Table(
                    name="NULLABLE",
                    columns=[
                        Column("ID", 1, "int", primary_key=1),
                        Column("VALUE", 2, "int"),
                    ],
                )
            ],
        )
    )
    dataset.validate(feedback=False)

    chunk = next(dataset["NULLABLE"].iterate_data_chunks(chunk_size=2))

    assert chunk.df["VALUE"].isna().tolist() == [True, True]


def make_date_timestamp_dataset(date_values, timestamp_values) -> Dataset:
    frame = DataFrame(
        {
            "ID": range(len(date_values)),
            "DATE": date_values,
            "TIMESTAMP": timestamp_values,
        }
    )
    dictionary = Dictionary(
        name="Date and timestamp validation",
        tables=[
            Table(
                name=TABLE_NAME,
                columns=[
                    Column("ID", 1, "int", primary_key=1),
                    Column("DATE", 2, "date"),
                    Column("TIMESTAMP", 3, "timestamp"),
                ],
            )
        ],
    )
    dataset = Dataset.create_from({TABLE_NAME: frame})
    dataset.import_dictionary(dictionary)
    return dataset


def test_validate_date_midnight_timestamp_and_preserve_timestamp_precision() -> None:
    dataset = make_date_timestamp_dataset(
        ["2022-01-01", "2022-02-01 00:00:00"],
        ["2022-01-01T00:00:00.123456789", "2022-02-01 00:00:00"],
    )

    dataset.validate(feedback=False)

    assert dataset.check()
    data = dataset[TABLE_NAME].data
    assert data["DATE"].dt.hour.tolist() == [0, 0]
    assert data["TIMESTAMP"].iloc[0].nanosecond == 789
    assert (
        dataset[TABLE_NAME].table_dictionary.get_column("DATE").datetime_format is None
    )
    assert (
        dataset[TABLE_NAME].table_dictionary.get_column("TIMESTAMP").datetime_format
        is None
    )


def test_validate_rejects_native_non_midnight_date_before_conversion() -> None:
    date_values = date_range("2022-01-01", periods=2, freq="D") + to_timedelta(
        ["0s", "1s"]
    )
    dataset = make_date_timestamp_dataset(
        date_values,
        ["2022-01-01 00:00:00", "2022-01-02 00:00:00"],
    )

    dataset.validate(feedback=False)

    with pytest.raises(DataIntegrityError):
        dataset.check()
    issues = dataset[TABLE_NAME].validator.issues
    assert any(issue.type == IssueType.TYPE_MISMATCH for issue in issues)
    assert dataset[TABLE_NAME].data["DATE"].iloc[1].second == 1


def test_validate_mixed_datetime_values_reports_every_invalid_row() -> None:
    date_values = Series(
        [
            "2022-01-01",
            "2022-02-01 00:00:00",
            "2022/03/01",
            Timestamp("2022-04-01 00:00:00"),
            Timestamp("2022-05-01 01:00:00"),
            "2022-06-01 00:00:00.000001",
            Timestamp("2022-07-01T00:00:00+00:00"),
            "not a date",
            "2022-09-01",
        ],
        dtype="object",
    )
    timestamp_values = Series(
        [
            "2022-01-01 00:00:00",
            "2022-02-01T01:02:03.123456789",
            Timestamp("2022-03-01 04:05:06"),
            "2022-04-01T00:00:00+0000",
            "2022-05-01",
            "not a timestamp",
            "2022/07/01 08:09:10",
            "2022-08-01 12:30:00",
            "2022-09-01T00:00:00.000000001",
        ],
        dtype="object",
    )
    dataset = Dataset.create_from(
        {
            TABLE_NAME: DataFrame(
                {"ID": range(1, 10), "DATE": date_values, "TIMESTAMP": timestamp_values}
            )
        }
    )
    dataset.import_dictionary(
        Dictionary(
            name="Mixed datetime values",
            tables=[
                Table(
                    name=TABLE_NAME,
                    columns=[
                        Column("ID", 1, "int", primary_key=1),
                        Column("DATE", 2, "date"),
                        Column("TIMESTAMP", 3, "timestamp"),
                    ],
                )
            ],
        )
    )

    dataset.validate(feedback=False)

    item = dataset[TABLE_NAME]
    date_issue = next(issue for issue in item.issues if issue.column == "DATE")
    timestamp_issue = next(
        issue for issue in item.issues if issue.column == "TIMESTAMP"
    )
    assert date_issue.type == IssueType.TYPE_MISMATCH
    assert date_issue.ranges == [Range(4, 7)]
    assert timestamp_issue.type == IssueType.TYPE_MISMATCH
    assert timestamp_issue.ranges == [Range(4, 6)]
    assert item.data["DATE"].iloc[4].hour == 1


def test_apply_dictionary_converts_valid_mixed_datetime_values() -> None:
    dataset = make_date_timestamp_dataset(
        Series(
            ["2022-01-01", "2022-02-01 00:00:00", Timestamp("2022-03-01")],
            dtype="object",
        ),
        Series(
            [
                "2022-01-01 00:00:00",
                "2022-02-01T01:02:03.123456789",
                Timestamp("2022-03-01 04:05:06"),
            ],
            dtype="object",
        ),
    )
    dataset.validate(feedback=False)

    assert dataset.check()
    data = dataset[TABLE_NAME].data
    assert str(data["DATE"].dtype) == "datetime64[ns]"
    assert str(data["TIMESTAMP"].dtype) == "datetime64[ns]"
    assert data["DATE"].dt.hour.tolist() == [0, 0, 0]
    assert data["TIMESTAMP"].iloc[1].nanosecond == 789
    assert data["TIMESTAMP"].iloc[2].hour == 4


@pytest.mark.parametrize(
    ("first_date", "conflicting_date", "first_timestamp", "conflicting_timestamp"),
    [
        (
            "13/04/2022",
            "04/13/2022",
            "13/04/2022 05:06:07",
            "04/13/2022 08:09:10",
        ),
        (
            "04/13/2022",
            "13/04/2022",
            "04/13/2022 08:09:10",
            "13/04/2022 05:06:07",
        ),
    ],
)
def test_validate_reports_mixed_day_first_and_month_first_formats(
    first_date: str,
    conflicting_date: str,
    first_timestamp: str,
    conflicting_timestamp: str,
) -> None:
    dataset = make_date_timestamp_dataset(
        Series(
            [Timestamp("2022-01-01"), first_date, conflicting_date, "03/04/2022"],
            dtype="object",
        ),
        Series(
            [
                Timestamp("2022-01-01 01:02:03"),
                first_timestamp,
                conflicting_timestamp,
                "03/04/2022 11:12:13",
            ],
            dtype="object",
        ),
    )
    date_order_format = "%d/%m/%Y" if first_date == "13/04/2022" else "%m/%d/%Y"
    timestamp_order_format = (
        "%d/%m/%Y %H:%M:%S" if first_date == "13/04/2022" else "%m/%d/%Y %H:%M:%S"
    )
    dataset[TABLE_NAME].table_dictionary.get_column(
        "DATE"
    ).datetime_format = date_order_format
    dataset[TABLE_NAME].table_dictionary.get_column(
        "TIMESTAMP"
    ).datetime_format = timestamp_order_format

    dataset.validate(feedback=False)

    item = dataset[TABLE_NAME]
    with pytest.raises(DataIntegrityError):
        item.check()

    date_issue = next(issue for issue in item.issues if issue.column == "DATE")
    timestamp_issue = next(
        issue for issue in item.issues if issue.column == "TIMESTAMP"
    )
    assert date_issue.type == IssueType.INCONSISTENT_DATE
    assert date_issue.ranges == [Range(1, 2)]
    assert timestamp_issue.type == IssueType.INCONSISTENT_DATE
    assert timestamp_issue.ranges == [Range(1, 2)]
    assert not any(
        issue.type == IssueType.TYPE_MISMATCH and issue.column in {"DATE", "TIMESTAMP"}
        for issue in item.issues
    )
    assert item.data["DATE"].iloc[2] == conflicting_date


def test_apply_dictionary_uses_established_order_for_ambiguous_values() -> None:
    dataset = make_date_timestamp_dataset(
        Series(["04/13/2022", "03/04/2022"], dtype="string"),
        Series(["04/13/2022 08:09:10", "03/04/2022 11:12:13"], dtype="string"),
    )
    dataset[TABLE_NAME].table_dictionary.get_column("DATE").datetime_format = "%d/%m/%Y"
    dataset[TABLE_NAME].table_dictionary.get_column(
        "TIMESTAMP"
    ).datetime_format = "%d/%m/%Y %H:%M:%S"

    dataset.validate(feedback=False)

    assert dataset.check()
    data = dataset[TABLE_NAME].data
    assert data["DATE"].dt.strftime("%Y-%m-%d").tolist() == [
        "2022-04-13",
        "2022-03-04",
    ]
    assert data["TIMESTAMP"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist() == [
        "2022-04-13 08:09:10",
        "2022-03-04 11:12:13",
    ]


def test_ambiguous_dates_do_not_establish_a_convention() -> None:
    dataset = make_date_timestamp_dataset(
        Series(["03/04/2022", "05/06/2022"], dtype="string"),
        Series(["03/04/2022 11:12:13", "05/06/2022 14:15:16"], dtype="string"),
    )

    dataset.validate(feedback=False)

    assert dataset.check()
    item = dataset[TABLE_NAME]
    assert item.validator._date_orders == {}
    assert not any(issue.type == IssueType.INCONSISTENT_DATE for issue in item.issues)
    assert item.data["DATE"].dt.strftime("%Y-%m-%d").tolist() == [
        "2022-04-03",
        "2022-06-05",
    ]


def test_date_convention_conflict_is_reported_across_csv_chunks(tmp_path) -> None:
    csv_path = tmp_path / f"{TABLE_NAME}.csv"
    DataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6],
            "DATE": [
                "13/04/2022",
                "14/04/2022",
                "04/13/2022",
                "15/04/2022",
                "05/14/2022",
                "16/04/2022",
            ],
            "TIMESTAMP": [
                "13/04/2022 05:06:07",
                "14/04/2022 05:06:07",
                "04/13/2022 08:09:10",
                "15/04/2022 05:06:07",
                "05/14/2022 08:09:10",
                "16/04/2022 05:06:07",
            ],
        }
    ).to_csv(csv_path, index=False)
    dataset = Dataset.create_from(csv_path)
    dataset.import_dictionary(
        Dictionary(
            name="Chunked mixed date conventions",
            tables=[
                Table(
                    name=TABLE_NAME,
                    columns=[
                        Column("ID", 1, "int", primary_key=1),
                        Column("DATE", 2, "date"),
                        Column("TIMESTAMP", 3, "timestamp"),
                    ],
                )
            ],
        )
    )

    dataset[TABLE_NAME].validate(chunk_size=2, feedback=False)

    item = dataset[TABLE_NAME]
    date_issue = next(
        issue
        for issue in item.issues
        if issue.type == IssueType.INCONSISTENT_DATE and issue.column == "DATE"
    )
    timestamp_issue = next(
        issue
        for issue in item.issues
        if issue.type == IssueType.INCONSISTENT_DATE and issue.column == "TIMESTAMP"
    )
    assert date_issue.ranges == [Range(0, 0), Range(2, 2), Range(4, 4)]
    assert timestamp_issue.ranges == [Range(0, 0), Range(2, 2), Range(4, 4)]
    assert date_issue.inspect(print_header=False)["DATE"].tolist() == [
        "13/04/2022",
        "04/13/2022",
        "05/14/2022",
    ]


@pytest.mark.parametrize("source", ["string_dataframe", "object_dataframe", "csv"])
def test_date_convention_detection_is_consistent_across_sources(
    source: str, tmp_path
) -> None:
    values = {
        "ID": [1, 2, 3],
        "DATE": ["13/04/2022", "14/04/2022", "04/13/2022"],
        "TIMESTAMP": [
            "13/04/2022 05:06:07",
            "14/04/2022 05:06:07",
            "04/13/2022 08:09:10",
        ],
    }
    if source == "csv":
        csv_path = tmp_path / f"{TABLE_NAME}.csv"
        DataFrame(values).to_csv(csv_path, index=False)
        dataset = Dataset.create_from(csv_path)
    else:
        dtype = "string" if source == "string_dataframe" else "object"
        frame = DataFrame(
            {
                "ID": values["ID"],
                "DATE": Series(values["DATE"], dtype=dtype),
                "TIMESTAMP": Series(values["TIMESTAMP"], dtype=dtype),
            }
        )
        dataset = Dataset.create_from({TABLE_NAME: frame})

    dataset.import_dictionary(
        Dictionary(
            name="Convention source parity",
            tables=[
                Table(
                    name=TABLE_NAME,
                    columns=[
                        Column("ID", 1, "int", primary_key=1),
                        Column("DATE", 2, "date"),
                        Column("TIMESTAMP", 3, "timestamp"),
                    ],
                )
            ],
        )
    )
    dataset[TABLE_NAME].validate(
        chunk_size=1 if source == "csv" else None,
        feedback=False,
    )

    issues = dataset[TABLE_NAME].issues
    assert [
        (issue.column, [(item.start, item.end) for item in issue.ranges])
        for issue in issues
        if issue.type == IssueType.INCONSISTENT_DATE
    ] == [
        ("DATE", [(0, 0), (2, 2)]),
        ("TIMESTAMP", [(0, 0), (2, 2)]),
    ]


def test_native_datetime_values_are_convention_neutral() -> None:
    dataset = make_date_timestamp_dataset(
        date_range("2022-04-13", periods=3, freq="D"),
        date_range("2022-04-13 05:00:00", periods=3, freq="h"),
    )

    dataset.validate(feedback=False)

    assert dataset.check()
    assert dataset[TABLE_NAME].validator._date_orders == {}
    assert not any(
        issue.type == IssueType.INCONSISTENT_DATE
        for issue in dataset[TABLE_NAME].issues
    )


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
