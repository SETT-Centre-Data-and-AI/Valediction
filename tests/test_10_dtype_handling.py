from collections import Counter
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from pandas import DataFrame
from pandas.testing import assert_frame_equal, assert_series_equal

from valediction import demo
from valediction.datasets.datasets import Dataset
from valediction.dictionary.model import Column, Dictionary, Table
from valediction.exceptions import DataIntegrityError
from valediction.integrity import Config
from valediction.validation import helpers as validation_helpers
from valediction.validation.helpers import (
    apply_data_types,
    invalid_mask_date,
    invalid_mask_datetime,
    invalid_mask_integer,
    invalid_mask_integer_out_of_range,
)

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


@pytest.mark.parametrize("representation", ["native", "declared", "csv", "mixed"])
def test_timestamp_formats_validate_and_convert(representation: str) -> None:
    dataset = get_mixed_dtype_dataset()
    item = dataset[TABLE_NAME]
    item.data.index = [17, 3, 17]
    datetime_format = "%d/%m/%Y %H:%M:%S"
    item.table_dictionary.get_column("TIMESTAMP").datetime_format = datetime_format
    expected = pd.Series(
        [
            pd.Timestamp("2023-08-22 22:13:07"),
            pd.Timestamp("2023-08-23 09:05:11"),
            pd.NaT,
        ],
        index=item.data.index,
        name="TIMESTAMP",
    )
    if representation == "native":
        values = expected.copy()
    else:
        values = pd.Series(
            [
                "2023-08-22 22:13:07"
                if representation == "csv"
                else "22/08/2023 22:13:07",
                "23/08/2023 09:05:11"
                if representation == "declared"
                else "2023-08-23 09:05:11",
                None,
            ],
            index=item.data.index,
            name="TIMESTAMP",
            dtype="string",
        )
    item.data["TIMESTAMP"] = values
    assert not invalid_mask_datetime(values, datetime_format).any()
    converted = apply_data_types(item.data.copy(), item.table_dictionary)
    assert_series_equal(converted["TIMESTAMP"], expected)

    for _ in range(2):
        dataset.validate(feedback=False)
        assert dataset.check()
        assert not item.issues
        assert_series_equal(item.data["TIMESTAMP"], expected)
        assert (
            item.table_dictionary.get_column("TIMESTAMP").datetime_format
            == datetime_format
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        "not-a-timestamp",
        "31/02/2023 09:05:11",
        "2023-02-31 09:05:11",
        "2023-02-31",
        "31/02/2023",
        "2023.08.22",
        "22/08/2023 extra",
        "08/23/2023 09:05:11",
        "NaT",
    ],
)
def test_timestamp_invalid_values_still_rejected(invalid_value: str) -> None:
    dataset = get_mixed_dtype_dataset()
    item = dataset[TABLE_NAME]
    datetime_format = "%d/%m/%Y %H:%M:%S"
    item.table_dictionary.get_column("TIMESTAMP").datetime_format = datetime_format
    item.data["TIMESTAMP"] = [invalid_value, "2023-08-23 09:05:11", None]
    assert invalid_mask_datetime(item.data["TIMESTAMP"], datetime_format).tolist() == [
        True,
        False,
        False,
    ]
    with pytest.raises(ValueError):
        apply_data_types(item.data.copy(), item.table_dictionary)
    dataset.validate(feedback=False)
    with pytest.raises(DataIntegrityError):
        dataset.check()
    assert len(item.issues) == 1
    issue = item.issues[0]
    expected_issue = (
        "INCONSISTENT_DATE"
        if invalid_value == "08/23/2023 09:05:11"
        else "TYPE_MISMATCH"
    )
    assert issue.type.name == expected_issue
    assert issue.column == "TIMESTAMP"
    assert [(span.start, span.end) for span in issue.ranges] == [(0, 0)]


@pytest.mark.parametrize(
    "datetime_format", [None, "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]
)
@pytest.mark.parametrize(
    "date_format",
    ["%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%m-%d-%Y"],
)
@pytest.mark.parametrize("date_only", [False, True], ids=["mixed", "date_only"])
def test_timestamp_dates_promote_to_midnight(
    datetime_format: str | None, date_format: str, date_only: bool
) -> None:
    dataset = get_mixed_dtype_dataset()
    item = dataset[TABLE_NAME]
    item.data.index = [17, 3, 17]
    item.table_dictionary.get_column("TIMESTAMP").datetime_format = datetime_format
    expected = pd.Series(
        [
            pd.Timestamp("2023-08-22"),
            pd.Timestamp("2023-08-23" if date_only else "2023-08-23 09:05:11"),
            pd.NaT,
        ],
        index=item.data.index,
        name="TIMESTAMP",
    )
    item.data["TIMESTAMP"] = pd.Series(
        [
            expected.iloc[0].strftime(date_format),
            expected.iloc[1].strftime(date_format)
            if date_only
            else "2023-08-23 09:05:11",
            pd.NA,
        ],
        index=item.data.index,
        dtype="string",
    )
    if datetime_format == "%d/%m/%Y %H:%M:%S" and date_format.startswith("%m"):
        with pytest.raises(ValueError, match="Inconsistent date"):
            apply_data_types(item.data.copy(), item.table_dictionary)
        dataset.validate(feedback=False)
        with pytest.raises(DataIntegrityError):
            dataset.check()
        assert item.issues[0].type.value == "InconsistentDate"
        assert [(span.start, span.end) for span in item.issues[0].ranges] == [
            (0, 1 if date_only else 0)
        ]
        return
    assert not invalid_mask_datetime(item.data["TIMESTAMP"], datetime_format).any()
    converted = apply_data_types(item.data.copy(), item.table_dictionary)
    assert_series_equal(converted["TIMESTAMP"], expected)
    for _ in range(2):
        dataset.validate(feedback=False)
        assert dataset.check()
        assert_series_equal(item.data["TIMESTAMP"], expected)
        if datetime_format is not None and (
            datetime_format.startswith("%d") or date_format.startswith("%Y")
        ):
            assert (
                item.table_dictionary.get_column("TIMESTAMP").datetime_format
                == datetime_format
            )


@pytest.mark.parametrize("declared", [False, True])
def test_timestamp_date_format_precedence(declared: bool) -> None:
    dataset = get_mixed_dtype_dataset()
    item = dataset[TABLE_NAME]
    datetime_format = "%m/%d/%Y" if declared else None
    item.table_dictionary.get_column("TIMESTAMP").datetime_format = datetime_format
    item.data["TIMESTAMP"] = [
        "01/02/2023",
        "08/22/2023" if declared else "22/08/2023",
        None,
    ]
    expected = pd.Series(
        [
            pd.Timestamp("2023-01-02" if declared else "2023-02-01"),
            pd.Timestamp("2023-08-22"),
            pd.NaT,
        ],
        name="TIMESTAMP",
    )
    dataset.validate(feedback=False)
    assert dataset.check()
    assert_series_equal(item.data["TIMESTAMP"], expected)


def test_timestamp_date_promotion_respects_config() -> None:
    dataset = get_mixed_dtype_dataset()
    item = dataset[TABLE_NAME]
    item.table_dictionary.get_column("DATE").datetime_format = "%Y-%m-%d"
    item.data["TIMESTAMP"] = ["2023-08-22", "22/08/2023", None]
    datetime_format = "%d/%m/%Y %H:%M:%S"
    item.table_dictionary.get_column("TIMESTAMP").datetime_format = datetime_format
    with Config() as config:
        config.date_formats = {
            "%d.%m.%Y": item.table_dictionary.get_column("DATE").data_type
        }
        assert invalid_mask_datetime(
            item.data["TIMESTAMP"], datetime_format
        ).tolist() == [True, True, False]
        with pytest.raises(ValueError):
            apply_data_types(item.data.copy(), item.table_dictionary)
        item.data["TIMESTAMP"] = ["22.08.2023", "23.08.2023", None]
        dataset.validate(feedback=False)
        assert dataset.check()
        assert_series_equal(
            item.data["TIMESTAMP"],
            pd.Series(
                [pd.Timestamp("2023-08-22"), pd.Timestamp("2023-08-23"), pd.NaT],
                name="TIMESTAMP",
            ),
        )


@pytest.mark.parametrize("month_first", [False, True])
def test_timestamp_configured_date_order(month_first: bool) -> None:
    with Config() as config:
        date_type = (
            get_mixed_dtype_dataset()[TABLE_NAME]
            .table_dictionary.get_column("DATE")
            .data_type
        )
        formats = ["%m/%d/%Y", "%d/%m/%Y"] if month_first else ["%d/%m/%Y", "%m/%d/%Y"]
        config.date_formats = {fmt: date_type for fmt in formats}
        dataset = get_mixed_dtype_dataset()
        item = dataset[TABLE_NAME]
        item.table_dictionary.get_column("DATE").datetime_format = "%Y-%m-%d"
        item.data["TIMESTAMP"] = ["01/02/2023", "02/01/2023", None]
        expected = pd.Series(
            [
                pd.Timestamp("2023-01-02" if month_first else "2023-02-01"),
                pd.Timestamp("2023-02-01" if month_first else "2023-01-02"),
                pd.NaT,
            ],
            name="TIMESTAMP",
        )
        dataset.validate(feedback=False)
        assert dataset.check()
        assert_series_equal(item.data["TIMESTAMP"], expected)


@pytest.mark.parametrize("mixed", [False, True])
def test_timestamp_native_dates_promote_to_midnight(mixed: bool) -> None:
    dataset = get_mixed_dtype_dataset()
    item = dataset[TABLE_NAME]
    item.table_dictionary.get_column("TIMESTAMP").datetime_format = "%d/%m/%Y %H:%M:%S"
    expected = pd.Series(
        [
            pd.Timestamp("2023-08-22"),
            pd.Timestamp("2023-08-23 09:05:11" if mixed else "2023-08-23"),
            pd.NaT,
        ],
        name="TIMESTAMP",
    )
    item.data["TIMESTAMP"] = [
        date(2023, 8, 22),
        expected.iloc[1] if mixed else date(2023, 8, 23),
        None,
    ]
    dataset.validate(feedback=False)
    assert dataset.check()
    assert_series_equal(item.data["TIMESTAMP"], expected)


def test_timestamp_declared_format_has_precedence() -> None:
    dataset = get_mixed_dtype_dataset()
    item = dataset[TABLE_NAME]
    datetime_format = "%Y-%d-%m %H:%M:%S"
    item.table_dictionary.get_column("TIMESTAMP").datetime_format = datetime_format
    item.data["TIMESTAMP"] = ["2023-08-09 22:13:07", "2023-08-23 09:05:11", None]
    assert not invalid_mask_datetime(item.data["TIMESTAMP"], datetime_format).any()
    converted = apply_data_types(item.data.copy(), item.table_dictionary)
    expected = pd.Series(
        [
            pd.Timestamp("2023-09-08 22:13:07"),
            pd.Timestamp("2023-08-23 09:05:11"),
            pd.NaT,
        ],
        name="TIMESTAMP",
    )
    assert_series_equal(converted["TIMESTAMP"], expected)


def test_timestamp_no_format_behavior_unchanged() -> None:
    dataset = get_mixed_dtype_dataset()
    item = dataset[TABLE_NAME]
    values = item.data["TIMESTAMP"].copy()
    assert not invalid_mask_datetime(values, None).any()
    with Config() as config:
        config.date_formats = {}
        assert invalid_mask_datetime(values, None).all()
    converted = apply_data_types(item.data.copy(), item.table_dictionary)
    assert_series_equal(converted["TIMESTAMP"], pd.to_datetime(values))


def test_date_accepts_compatible_representations() -> None:
    dataset = get_mixed_dtype_dataset()
    item = dataset[TABLE_NAME]
    item.table_dictionary.get_column("DATE").datetime_format = "%d/%m/%Y"
    item.data["DATE"] = ["22/08/2023", "2023-08-23", None]
    assert invalid_mask_date(item.data["DATE"], "%d/%m/%Y").tolist() == [
        False,
        False,
        False,
    ]
    converted = apply_data_types(item.data.copy(), item.table_dictionary)
    assert_series_equal(
        converted["DATE"],
        pd.Series(
            [pd.Timestamp("2023-08-22"), pd.Timestamp("2023-08-23"), pd.NaT],
            name="DATE",
        ),
    )


@pytest.mark.parametrize(
    "values,expected_values",
    [
        (pd.Series([1, None, -2], dtype="Int64"), [1, None, -2]),
        (pd.Series([1.0, None, -2.0]), [1, None, -2]),
        (
            pd.Series([9007199254740993, None, -9007199254740993], dtype=object),
            [9007199254740993, None, -9007199254740993],
        ),
        (
            pd.Series(["9007199254740993", "1.0", None], dtype="string"),
            [9007199254740993, 1, None],
        ),
        (
            pd.Series(
                ["9223372036854775807.0", "-9223372036854775808", "1e2"], dtype="string"
            ),
            [9223372036854775807, -9223372036854775808, 100],
        ),
        (
            pd.Series(
                ["9.223372036854775807e18", "-9.223372036854775808e18", None],
                dtype="string",
            ),
            [9223372036854775807, -9223372036854775808, None],
        ),
    ],
    ids=[
        "nullable",
        "float_null",
        "object_null",
        "mixed_decimal",
        "boundaries",
        "scientific_boundaries",
    ],
)
def test_integer_fidelity(values: pd.Series, expected_values: list) -> None:
    dataset = get_mixed_dtype_dataset()
    item = dataset[TABLE_NAME]
    item.data.index = [17, 3, 17]
    item.table_dictionary.add_column(Column(name="VALUE", order=6, data_type="int"))
    values = values.copy()
    values.index = item.data.index
    item.data["VALUE"] = values
    expected = pd.Series(
        expected_values, dtype="Int64", index=item.data.index, name="VALUE"
    )
    assert not invalid_mask_integer(values).any()
    converted = apply_data_types(item.data.copy(), item.table_dictionary)
    assert_series_equal(converted["VALUE"], expected)
    for _ in range(2):
        dataset.validate(feedback=False)
        assert dataset.check()
        assert_series_equal(item.data["VALUE"], expected)


@pytest.mark.parametrize("cache_limit", [0, 2])
def test_integer_cache_saturation_preserves_results(
    monkeypatch: pytest.MonkeyPatch, cache_limit: int
) -> None:
    monkeypatch.setattr(validation_helpers, "_INTEGER_CACHE_LIMIT", cache_limit)
    matcher = Mock(wraps=validation_helpers._INTEGER_TEXT)
    monkeypatch.setattr(validation_helpers, "_INTEGER_TEXT", matcher)
    tokens = [
        "9007199254740993",
        "bad",
        "9223372036854775807.0",
        "-9223372036854775808",
        "9.223372036854775807e18",
        "1e2",
        "1.0000000000005",
        "1.000000000002",
        "9223372036854775808",
        "9007199254740993.5",
        None,
    ]
    results = [
        9007199254740993,
        None,
        9223372036854775807,
        -9223372036854775808,
        9223372036854775807,
        100,
        1,
        None,
        None,
        None,
        None,
    ]
    column = pd.Series(
        tokens * 2, index=[17] * (len(tokens) * 2), name="VALUE", dtype="string"
    )
    expected = pd.Series(
        results * 2, index=column.index, name=column.name, dtype="Int64"
    )
    expected_calls = {
        token: 1 if cache_limit and position < cache_limit else 2
        for position, token in enumerate(tokens)
        if token is not None
    }
    for _ in range(2):
        matcher.reset_mock()
        parsed = validation_helpers._parse_integer(column, errors="coerce")
        assert_series_equal(parsed, expected)
        assert (
            Counter(call.args[0] for call in matcher.fullmatch.call_args_list)
            == expected_calls
        )


def test_integer_cache_stops_at_default_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    assert validation_helpers._INTEGER_CACHE_LIMIT == 65_536
    base = 9007199254740993
    limit = validation_helpers._INTEGER_CACHE_LIMIT
    numbers = list(range(base, base + limit + 2))
    cached_text = str(numbers[0])
    uncached_text = str(numbers[limit])
    calls = Counter()
    original_match = validation_helpers._INTEGER_TEXT.fullmatch

    def counted_match(text: str):
        if text in (cached_text, uncached_text):
            calls[text] += 1
        return original_match(text)

    monkeypatch.setattr(
        validation_helpers, "_INTEGER_TEXT", SimpleNamespace(fullmatch=counted_match)
    )
    column = pd.Series(
        [str(number) for number in numbers] + [cached_text, uncached_text, None],
        name="VALUE",
        dtype="string",
    )
    parsed = validation_helpers._parse_integer(column, errors="raise")
    expected = pd.Series(
        numbers + [numbers[0], numbers[limit], None], name="VALUE", dtype="Int64"
    )
    assert_series_equal(parsed, expected)
    assert calls == {cached_text: 1, uncached_text: 2}


@pytest.mark.parametrize(
    "invalid_value", ["bad", "9223372036854775808", "9007199254740993.5"]
)
def test_integer_cache_saturation_still_raises(
    monkeypatch: pytest.MonkeyPatch, invalid_value: str
) -> None:
    monkeypatch.setattr(validation_helpers, "_INTEGER_CACHE_LIMIT", 2)
    column = pd.Series(["1", "2", invalid_value, invalid_value], dtype="string")
    assert invalid_mask_integer(column).tolist() == [False, False, True, True]
    with pytest.raises(ValueError, match="integer-equivalent"):
        validation_helpers._parse_integer(column, errors="raise")


@pytest.mark.parametrize("tolerance,expected", [(1e-12, 1), (0, None)])
def test_integer_uncached_tolerance(
    monkeypatch: pytest.MonkeyPatch, tolerance: float, expected: int | None
) -> None:
    monkeypatch.setattr(validation_helpers, "_INTEGER_CACHE_LIMIT", 0)
    column = pd.Series(["1.0000000000005"] * 2, dtype="string")
    parsed = validation_helpers._parse_integer(
        column, errors="coerce", tolerance=tolerance
    )
    assert_series_equal(parsed, pd.Series([expected] * 2, dtype="Int64"))


@pytest.mark.parametrize(
    "value",
    [
        "9007199254740993.5",
        "9223372036854775808",
        "-9223372036854775809",
        "1e100",
        "1e99999999999999999999999",
        "NaN",
        "Infinity",
        "1_000",
        "bad",
    ],
)
def test_integer_invalid_values_rejected(value: str) -> None:
    dataset = get_mixed_dtype_dataset()
    item = dataset[TABLE_NAME]
    item.data["INT"] = [value, "2", "3"]
    assert invalid_mask_integer(item.data["INT"]).tolist() == [True, False, False]
    with pytest.raises(ValueError):
        apply_data_types(item.data.copy(), item.table_dictionary)
    dataset.validate(feedback=False)
    with pytest.raises(DataIntegrityError):
        dataset.check()


def test_integer_int4_equivalent_representations() -> None:
    values = pd.Series(
        [
            "2147483647.0",
            "-2147483648.0",
            "00000000001",
            "1e10",
            "-2147483649.0",
            "3.14",
            None,
        ]
    )
    assert invalid_mask_integer_out_of_range(values).tolist() == [
        False,
        False,
        False,
        True,
        True,
        False,
        False,
    ]


def test_integer_int4_validation_issue_ranges() -> None:
    dataset = get_mixed_dtype_dataset()
    item = dataset[TABLE_NAME]
    item.data["INT"] = ["2147483647.0", "1e10", "3.14"]
    with Config() as config:
        config.allow_bigint = False
        dataset.validate(feedback=False)
    assert not item.validated
    issues = {issue.type.name: issue for issue in item.issues}
    assert set(issues) == {"TYPE_MISMATCH", "INTEGER_OUT_OF_RANGE"}
    assert [(span.start, span.end) for span in issues["TYPE_MISMATCH"].ranges] == [
        (2, 2)
    ]
    assert [
        (span.start, span.end) for span in issues["INTEGER_OUT_OF_RANGE"].ranges
    ] == [(1, 1)]


@pytest.mark.parametrize("dtype", ["UInt64", "uint64"])
def test_integer_unsigned_overflow_does_not_wrap(dtype: str) -> None:
    values = pd.Series([9223372036854775807, 9223372036854775808], dtype=dtype)
    assert invalid_mask_integer(values).tolist() == [False, True]


def test_timestamp_mixed_offsets_remain_accepted() -> None:
    dataset = get_mixed_dtype_dataset()
    item = dataset[TABLE_NAME]
    item.table_dictionary.get_column(
        "TIMESTAMP"
    ).datetime_format = "%Y-%m-%dT%H:%M:%S%z"
    values = ["2023-01-22T00:00:00+00:00", "2023-08-23T09:05:11+01:00", None]
    item.data["TIMESTAMP"] = values
    expected = pd.Series(
        [pd.Timestamp(value) if value else pd.NaT for value in values],
        dtype=object,
        name="TIMESTAMP",
    )
    dataset.validate(feedback=False)
    assert dataset.check()
    assert_series_equal(item.data["TIMESTAMP"], expected)


@pytest.mark.parametrize("configured_type", ["DATE", "TIMESTAMP"])
def test_timestamp_conversion_respects_config_formats(configured_type: str) -> None:
    dataset = get_mixed_dtype_dataset()
    item = dataset[TABLE_NAME]
    item.table_dictionary.get_column("DATE").datetime_format = "%Y-%m-%d"
    with Config() as config:
        config.date_formats = {}
        assert invalid_mask_datetime(item.data["TIMESTAMP"], None).all()
        with pytest.raises(ValueError):
            apply_data_types(item.data.copy(), item.table_dictionary)
        config.date_formats = {
            "%Y-%m-%d": item.table_dictionary.get_column(configured_type).data_type
        }
        assert invalid_mask_datetime(item.data["TIMESTAMP"], None).all()
        item.data["TIMESTAMP"] = ["2023-08-22", "2023-08-23", None]
        assert not invalid_mask_datetime(item.data["TIMESTAMP"], None).any()
        converted = apply_data_types(item.data.copy(), item.table_dictionary)
        assert_series_equal(
            converted["TIMESTAMP"],
            pd.Series(
                [pd.Timestamp("2023-08-22"), pd.Timestamp("2023-08-23"), pd.NaT],
                name="TIMESTAMP",
            ),
        )


def test_timestamp_fractional_formats_preserve_index_and_values() -> None:
    dataset = get_mixed_dtype_dataset()
    item = dataset[TABLE_NAME]
    item.data.index = [17, 3, 17]
    item.table_dictionary.get_column("TIMESTAMP").datetime_format = "%d/%m/%Y %H:%M:%S"
    item.data["TIMESTAMP"] = [
        "22/08/2023 00:00:00",
        "2023-08-23 09:05:11.123456789",
        "2023-08-24T09:05:11.123456",
    ]
    expected = pd.Series(
        [
            pd.Timestamp("2023-08-22"),
            pd.Timestamp("2023-08-23 09:05:11.123456789"),
            pd.Timestamp("2023-08-24 09:05:11.123456"),
        ],
        index=item.data.index,
        name="TIMESTAMP",
    )
    dataset.validate(feedback=False)
    assert dataset.check()
    assert_series_equal(item.data["TIMESTAMP"], expected)


if __name__ == "__main__":
    pytest.main([__file__])
