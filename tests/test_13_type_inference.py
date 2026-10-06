import pytest
from pandas import DataFrame, NaT, Series, Timestamp, to_datetime
from pandas.testing import assert_series_equal

from valediction.data_types.data_type_helpers import (
    _DatetimeResolver,
    _parse_offset_datetimes,
    infer_datetime_format,
)
from valediction.data_types.data_types import DataType
from valediction.data_types.type_inference import TypeInferer
from valediction.integrity import Config, reset_default_config
from valediction.progress import Progress
from valediction.validation.helpers import invalid_mask_date


@pytest.fixture(autouse=True)
def _isolate_global_default():  # noqa
    reset_default_config()
    yield
    reset_default_config()


@pytest.mark.filterwarnings("error::FutureWarning")
def test_cached_datetime_parse_handles_mixed_timezone_offsets() -> None:
    inferer = TypeInferer(dayfirst=True)
    values = Series(
        [
            "2021-11-27T16:56:00+00:00",
            "2021-06-27T17:56:00+01:00",
        ],
        dtype="string",
    )

    ok, has_time = inferer._parse_with_cached_format(values, "%Y-%m-%dT%H:%M:%S%z")

    assert ok.tolist() == [True, True]
    assert has_time.tolist() == [True, True]


@pytest.mark.filterwarnings("error::FutureWarning")
@pytest.mark.parametrize("fraction", ["", ".123456789"])
def test_offset_parsing_preserves_values_and_positions(fraction: str) -> None:
    fmt = "%Y-%m-%dT%H:%M:%S" + (".%f" if fraction else "") + "%z"
    timestamps = [
        f"2023-08-22T00:00:00{fraction}{offset}"
        for offset in ["+05:45", "-03:30", "Z", "+0545"]
    ]
    values = Series(
        timestamps + [None, "invalid"],
        index=[17, 3, 17, 2, 9, 9],
        name="TIMESTAMP",
        dtype="string",
    )
    expected = Series(
        [Timestamp(value) for value in timestamps] + [NaT, NaT],
        index=values.index,
        name=values.name,
        dtype=object,
    )

    resolver = _DatetimeResolver(DataType.TIMESTAMP, fmt)
    assert resolver.update(values).tolist() == [False] * 5 + [True]
    actual = resolver.parse(values, errors="coerce")

    assert_series_equal(actual, expected)
    assert [value.isoformat() for value in actual.iloc[:4]] == [
        value.isoformat() for value in expected.iloc[:4]
    ]
    with pytest.raises(ValueError, match="Values do not match"):
        _DatetimeResolver(DataType.TIMESTAMP, fmt).parse(values, errors="raise")


@pytest.mark.filterwarnings("error::FutureWarning")
@pytest.mark.parametrize(
    "values",
    [
        [],
        [None, None],
        ["2023-08-22T00:00:00+05:45", None],
        ["2023-08-22T00:00:00Z", "2023-08-23T00:00:00+0000"],
    ],
)
def test_offset_parsing_keeps_homogeneous_dtype(values: list) -> None:
    source = Series(values, dtype="string", name="TIMESTAMP")
    fmt = "%Y-%m-%dT%H:%M:%S%z"
    assert_series_equal(
        _parse_offset_datetimes(source, fmt),
        to_datetime(source, format=fmt, errors="coerce", utc=False),
    )


@pytest.mark.filterwarnings("error::FutureWarning")
def test_custom_offset_format_preserves_nanoseconds() -> None:
    fmt = "%z %d/%m/%Y %H:%M:%S.%f"
    values = Series(
        [
            "+0545 22/08/2023 00:00:00.123456789",
            "-0330 23/08/2023 09:05:11.987654321",
            None,
        ],
        name="TIMESTAMP",
        dtype="string",
    )
    expected = Series(
        [
            Timestamp("2023-08-22T00:00:00.123456789+05:45"),
            Timestamp("2023-08-23T09:05:11.987654321-03:30"),
            NaT,
        ],
        name=values.name,
        dtype=object,
    )
    with Config() as config:
        config.date_formats = {fmt: DataType.TIMESTAMP}
        actual = _DatetimeResolver(DataType.TIMESTAMP).parse(values, errors="raise")
    assert_series_equal(actual, expected)
    assert [value.isoformat() for value in actual.iloc[:2]] == [
        value.isoformat() for value in expected.iloc[:2]
    ]


@pytest.mark.filterwarnings("error::FutureWarning")
def test_offset_parsing_keeps_naive_values_and_local_midnight() -> None:
    values = Series(
        ["2023-08-22T00:00:00", "2023-08-22T00:00:00+05:45", None],
        name="TIMESTAMP",
        dtype="string",
    )
    expected = Series(
        [Timestamp(values.iloc[0]), Timestamp(values.iloc[1]), NaT],
        name=values.name,
        dtype=object,
    )
    assert_series_equal(
        _DatetimeResolver(DataType.TIMESTAMP).parse(values, errors="raise"), expected
    )
    assert infer_datetime_format(values.iloc[:2]) == "%Y-%m-%dT%H:%M:%S"
    inferer = TypeInferer(dayfirst=True, progress=Progress(enabled=False))
    inferer.update_with_chunk(DataFrame({"VALUE": values}))
    inferer.finalise()
    assert inferer.states["VALUE"].data_type is DataType.DATE


def test_datetime_inference_scans_after_unique_candidate() -> None:
    values = Series(["22/08/2023"] * 100 + ["08/23/2023"], dtype="string")
    assert infer_datetime_format(values) is None


@pytest.mark.parametrize("month_first", [False, True])
def test_datetime_inference_resolves_ambiguity_from_config(month_first: bool) -> None:
    formats = ["%m/%d/%Y", "%d/%m/%Y"] if month_first else ["%d/%m/%Y", "%m/%d/%Y"]
    with Config() as config:
        config.date_formats = {fmt: DataType.DATE for fmt in formats}
        assert (
            infer_datetime_format(Series(["01/02/2023"], dtype="string")) == formats[0]
        )


def test_datetime_inference_accepts_compatible_spellings() -> None:
    values = Series(["2023-08-22", "22/08/2023", "23-08-2023"], dtype="string")
    assert infer_datetime_format(values) == "%d/%m/%Y"


def test_datetime_tie_break_ignores_nonmatching_formats() -> None:
    with Config() as config:
        config.date_formats = {
            "%d/%m/%Y": DataType.DATE,
            "%m/%d/%Y %H:%M:%S": DataType.TIMESTAMP,
            "%d/%m/%Y %H:%M:%S": DataType.TIMESTAMP,
        }
        values = Series(["01/02/2023 12:00:00"], dtype="string")
        assert infer_datetime_format(values) == "%m/%d/%Y %H:%M:%S"
        parsed = _DatetimeResolver(DataType.TIMESTAMP).parse(values, errors="raise")
        assert parsed.iloc[0] == Timestamp("2023-01-02 12:00:00")


@pytest.mark.parametrize(
    "values",
    [
        ["2023-08-22T00:00:00.000000001", "2023-08-23", None],
        ["2023-08-22", "2023-08-23T00:00:00.000000001", None],
    ],
)
def test_datetime_inference_keeps_timestamp_across_chunks(values: list) -> None:
    inferer = TypeInferer(dayfirst=True, progress=Progress(enabled=False))
    for value in values:
        inferer.update_with_chunk(DataFrame({"VALUE": [value]}))
    inferer.finalise()
    assert inferer.states["VALUE"].data_type is DataType.TIMESTAMP
    assert inferer.states["VALUE"].cached_datetime_format is not None


@pytest.mark.parametrize("native", [False, True])
def test_date_rejects_nanosecond_time(native: bool) -> None:
    value = Timestamp("2023-08-22T00:00:00.000000001")
    values = Series([value if native else value.isoformat(), None])
    assert invalid_mask_date(values, "%Y-%m-%dT%H:%M:%S.%f").tolist() == [True, False]


@pytest.mark.parametrize(
    "fmt,value", [("%d.%m.%Y", "22.08.2023"), ("%d %B %Y", "22 August 2023")]
)
def test_datetime_inference_custom_formats(fmt: str, value: str) -> None:
    with Config() as config:
        config.date_formats = {fmt: DataType.DATE}
        inferer = TypeInferer(dayfirst=False, progress=Progress(enabled=False))
        inferer.update_with_chunk(DataFrame({"VALUE": [value]}))
        inferer.finalise()
        assert inferer.states["VALUE"].data_type is DataType.DATE
        assert inferer.states["VALUE"].cached_datetime_format == fmt


def test_datetime_direct_conversion_conflict_preserves_ambiguous_mask() -> None:
    values = Series(
        ["22/08/2023", "01/02/2023", "08/23/2023", "2023-08-24", None],
        index=[9, 1, 9, 2, 1],
    )
    for dtype in (DataType.DATE, DataType.TIMESTAMP):
        resolver = _DatetimeResolver(dtype)
        parsed = resolver.parse(values, errors="coerce")
        assert parsed.isna().tolist() == [True, False, True, False, True]
        assert parsed.index.equals(values.index)
        with pytest.raises(ValueError, match="Inconsistent date"):
            _DatetimeResolver(dtype).parse(values, errors="raise")


def test_unparseable_dateish_values_fall_back_to_text() -> None:
    inferer = TypeInferer(
        dayfirst=True,
        debug=True,
        progress=Progress(enabled=False),
    )
    df = DataFrame(
        {
            "MAYBE_DATE": [
                "2021-not-a-date",
                "2021/13/40",
                "not/T/date",
            ]
        }
    )

    inferer.update_with_chunk(df)
    data_type, length = inferer.states["MAYBE_DATE"].final_data_type_and_length()

    assert data_type is DataType.TEXT
    assert length == len("2021-not-a-date")


@pytest.mark.parametrize(
    ("allow_bigint", "expected_type", "expected_length"),
    [
        (True, DataType.INTEGER, None),
        (False, DataType.TEXT, len("2147483649")),
    ],
    ids=["allow_bigint", "disallow_bigint"],
)
def test_bigint_behavior_is_consistent(
    allow_bigint: bool,
    expected_type: DataType,
    expected_length: int | None,
) -> None:
    inferer = TypeInferer(
        dayfirst=True,
        progress=Progress(enabled=False),
    )
    df = DataFrame({"BIGS": [2147483648, 2147483649]})

    with Config() as config:
        config.allow_bigint = allow_bigint
        inferer.update_with_chunk(df)

    data_type, length = inferer.states["BIGS"].final_data_type_and_length()

    assert data_type is expected_type
    assert length == expected_length
