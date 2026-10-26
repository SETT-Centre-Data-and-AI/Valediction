import pytest
from pandas import NA, DataFrame, Series

from valediction.data_types.data_types import DataType
from valediction.datasets.datasets import Dataset
from valediction.dictionary.model import Column, Dictionary, Table
from valediction.integrity import get_config, reset_default_config
from valediction.validation.issues import IssueType, Range

TABLE_NAME = "FULL_VALIDATION"


@pytest.fixture(autouse=True)
def _isolate_global_default():
    reset_default_config()
    yield
    reset_default_config()


def _configure_validation() -> None:
    config = get_config()
    config.allow_bigint = False
    config.enforce_no_null_columns = True
    config.forbidden_characters = ["!"]
    config.date_formats["%m/%d/%Y %H:%M:%S"] = DataType.TIMESTAMP


def _create_frame() -> DataFrame:
    values = {
        "PK_TEXT": ["A !long", "A !long", "K E", "D", "E", "F"],
        "PK_NUM": ["1", "1", "2", NA, "4", "5"],
        "VALUE_INT": ["2147483648", "not-an-int", "3", "4", "5", "6"],
        "VALUE_FLOAT": ["1.5", "2.5", "3.5", "4.5", "not-a-float", "6.5"],
        "EVENT_DATE": [
            "13/04/2022",
            "04/13/2022",
            "03/04/2022",
            "not-a-date",
            "2022-04-14 12:30:00",
            "2022-04-15 00:00:00",
        ],
        "EVENT_TS": [
            "13/04/2022 05:06:07",
            "04/13/2022 08:09:10",
            "03/04/2022 07:08:09",
            "not-a-timestamp",
            "2022-04-14 12:30:00",
            "2022-04-15 00:00:00",
        ],
        "NOTE": ["ok", "ok", "bad!", "ok", "ok", NA],
        "ALL_NULL": ["", "null", "NONE", NA, " ", ""],
        "EXTRA": ["x"] * 6,
    }
    return DataFrame(
        {name: Series(column, dtype="string") for name, column in values.items()}
    )


def _create_dictionary() -> Dictionary:
    return Dictionary(
        name="Full validation matrix",
        tables=[
            Table(
                name=TABLE_NAME,
                columns=[
                    Column("PK_TEXT", 1, "text", length=3, primary_key=1),
                    Column("PK_NUM", 2, "int", primary_key=2),
                    Column("VALUE_INT", 3, "int"),
                    Column("VALUE_FLOAT", 4, "float"),
                    Column("EVENT_DATE", 5, "date"),
                    Column("EVENT_TS", 6, "timestamp"),
                    Column("NOTE", 7, "text", length=3, required=True),
                    Column("ALL_NULL", 8, "text", length=1),
                    Column("MISSING_OPTIONAL", 9, "text", length=100),
                ],
            )
        ],
    )


def _expected_issues() -> dict[tuple[IssueType, str | None], list[Range]]:
    return {
        (IssueType.MISSING_COLUMN, "MISSING_OPTIONAL"): [],
        (IssueType.EXTRA_COLUMN, "EXTRA"): [],
        (IssueType.FULLY_NULL_COLUMN, "ALL_NULL"): [],
        (IssueType.REQUIRED_NULL, "NOTE"): [Range(5, 5)],
        (IssueType.PK_COLLISION, None): [Range(0, 1)],
        (IssueType.PK_NULL, None): [Range(3, 3)],
        (IssueType.PK_WHITESPACE, None): [Range(0, 2)],
        (IssueType.TYPE_MISMATCH, "VALUE_INT"): [Range(1, 1)],
        (IssueType.TYPE_MISMATCH, "VALUE_FLOAT"): [Range(4, 4)],
        (IssueType.TYPE_MISMATCH, "EVENT_DATE"): [Range(3, 4)],
        (IssueType.TYPE_MISMATCH, "EVENT_TS"): [Range(3, 3)],
        (IssueType.INCONSISTENT_DATE, "EVENT_DATE"): [Range(0, 1)],
        (IssueType.INCONSISTENT_DATE, "EVENT_TS"): [Range(0, 1)],
        (IssueType.TEXT_TOO_LONG, "PK_TEXT"): [Range(0, 1)],
        (IssueType.TEXT_TOO_LONG, "NOTE"): [Range(2, 2)],
        (IssueType.FORBIDDEN_CHARACTER, "PK_TEXT"): [Range(0, 1)],
        (IssueType.FORBIDDEN_CHARACTER, "NOTE"): [Range(2, 2)],
        (IssueType.INTEGER_OUT_OF_RANGE, "VALUE_INT"): [Range(0, 0)],
    }


def _assert_full_issue_matrix(dataset: Dataset) -> None:
    item = dataset[TABLE_NAME]
    observed = {(issue.type, issue.column): issue.ranges for issue in item.issues}
    expected = _expected_issues()

    assert {issue_type for issue_type, _ in expected} == set(IssueType)
    assert observed == expected

    date_issue = next(
        issue
        for issue in item.issues
        if issue.type == IssueType.INCONSISTENT_DATE and issue.column == "EVENT_DATE"
    )
    assert date_issue.inspect(print_header=False)["EVENT_DATE"].tolist() == [
        "13/04/2022",
        "04/13/2022",
    ]


def test_full_validation_issue_matrix_dataframe() -> None:
    _configure_validation()
    dataset = Dataset.create_from({TABLE_NAME: _create_frame()})
    dataset.import_dictionary(_create_dictionary())

    dataset[TABLE_NAME].validate(feedback=False)

    _assert_full_issue_matrix(dataset)


@pytest.mark.parametrize("chunk_size", [None, 1, 2, 20])
def test_full_validation_issue_matrix_csv(tmp_path, chunk_size: int | None) -> None:
    _configure_validation()
    csv_path = tmp_path / f"{TABLE_NAME}.csv"
    _create_frame().to_csv(csv_path, index=False)
    dataset = Dataset.create_from(csv_path)
    dataset.import_dictionary(_create_dictionary())

    dataset[TABLE_NAME].validate(chunk_size=chunk_size, feedback=False)

    _assert_full_issue_matrix(dataset)
