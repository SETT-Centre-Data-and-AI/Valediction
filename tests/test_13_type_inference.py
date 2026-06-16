import pytest
from pandas import DataFrame, Series

from valediction.data_types.data_types import DataType
from valediction.data_types.type_inference import TypeInferer
from valediction.integrity import Config, reset_default_config
from valediction.progress import Progress


@pytest.fixture(autouse=True)
def _isolate_global_default():  # noqa
    reset_default_config()
    yield
    reset_default_config()


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
