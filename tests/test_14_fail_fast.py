import pytest
from pandas import DataFrame

from valediction.convenience import validate
from valediction.datasets.datasets import Dataset
from valediction.dictionary.model import Column, Dictionary, Table
from valediction.exceptions import DataIntegrityError
from valediction.integrity import reset_default_config
from valediction.progress import Progress
from valediction.validation.issues import IssueType, Range


@pytest.fixture(autouse=True)
def _isolate_global_default():
    reset_default_config()
    yield
    reset_default_config()


def _frame(keys: list[str], values: list[str]) -> DataFrame:
    return DataFrame({"PK_TEXT": keys, "VALUE_INT": values})


def _dictionary(table_names: list[str]) -> Dictionary:
    return Dictionary(
        name="Fail-fast validation",
        tables=[
            Table(
                name=table_name,
                columns=[
                    Column("PK_TEXT", 1, "text", length=3, primary_key=1),
                    Column("VALUE_INT", 2, "int"),
                ],
            )
            for table_name in table_names
        ],
    )


def _assert_first_check_error(error: DataIntegrityError) -> None:
    assert "more issues may exist" in str(error).lower()


def test_item_validate_fail_fast_keeps_first_check_matches() -> None:
    dataset = Dataset.create_from(
        {"FIRST": _frame(["A B", "C D", "EFGH"], ["1", "nope", "bad"])}
    )
    dataset.import_dictionary(_dictionary(["FIRST"]))
    item = dataset["FIRST"]

    with pytest.raises(DataIntegrityError) as exc_info:
        item.validate(feedback=False, fail_fast=True)

    _assert_first_check_error(exc_info.value)
    assert [(issue.type, issue.column, issue.ranges) for issue in item.issues] == [
        (IssueType.PK_WHITESPACE, None, [Range(0, 1)])
    ]
    assert item.issues[0].inspect(print_header=False)["PK_TEXT"].tolist() == [
        "A B",
        "C D",
    ]
    assert not item.validated


def test_item_validate_fail_fast_keeps_progress_incomplete(monkeypatch) -> None:
    finish_calls = []
    original_finish = Progress.finish

    def capture_finish(self, *args, **kwargs):
        finish_calls.append(kwargs)
        return original_finish(self, *args, **kwargs)

    monkeypatch.setattr(Progress, "finish", capture_finish)
    dataset = Dataset.create_from(
        {"FIRST": _frame(["A B", "C D", "EFGH"], ["1", "nope", "bad"])}
    )
    dataset.import_dictionary(_dictionary(["FIRST"]))

    with pytest.raises(DataIntegrityError):
        dataset["FIRST"].validate(feedback=True, fail_fast=True)

    assert len(finish_calls) == 1
    assert finish_calls[0]["postfix"].startswith("Failed fast with issues (")
    assert finish_calls[0]["complete"] is False


def test_item_validate_default_still_collects_all_issues() -> None:
    dataset = Dataset.create_from(
        {"FIRST": _frame(["A B", "C D", "EFGH"], ["1", "nope", "bad"])}
    )
    dataset.import_dictionary(_dictionary(["FIRST"]))
    item = dataset["FIRST"]

    item.validate(feedback=False)

    assert {issue.type for issue in item.issues} == {
        IssueType.PK_WHITESPACE,
        IssueType.TEXT_TOO_LONG,
        IssueType.TYPE_MISMATCH,
    }
    assert {issue.type: issue.ranges for issue in item.issues} == {
        IssueType.PK_WHITESPACE: [Range(0, 1)],
        IssueType.TEXT_TOO_LONG: [Range(2, 2)],
        IssueType.TYPE_MISMATCH: [Range(1, 2)],
    }


def test_item_validate_fail_fast_stops_before_later_csv_chunks(tmp_path) -> None:
    csv_path = tmp_path / "FIRST.csv"
    _frame(["A B", "C D", "EFGH", "IJKL"], ["1", "2", "bad", "also-bad"]).to_csv(
        csv_path, index=False
    )
    dataset = Dataset.create_from(csv_path)
    dataset.import_dictionary(_dictionary(["FIRST"]))

    with pytest.raises(DataIntegrityError) as exc_info:
        dataset["FIRST"].validate(chunk_size=2, feedback=False, fail_fast=True)

    _assert_first_check_error(exc_info.value)


def test_dataset_validate_fail_fast_stops_after_first_failing_item() -> None:
    dataset = Dataset.create_from(
        {
            "FIRST": _frame(["A B", "C D"], ["1", "2"]),
            "SECOND": _frame(["AB", "CD"], ["1", "2"]),
        }
    )
    dataset.import_dictionary(_dictionary(["FIRST", "SECOND"]))

    with pytest.raises(DataIntegrityError) as exc_info:
        dataset.validate(feedback=False, fail_fast=True)

    error = exc_info.value
    _assert_first_check_error(error)
    assert [(issue.type, issue.column) for issue in dataset.issues] == [
        (IssueType.PK_WHITESPACE, None)
    ]
    assert not dataset["SECOND"].validated
    assert dataset["SECOND"].validator is None


def test_convenience_validate_fail_fast_exposes_partial_dataset() -> None:
    with pytest.raises(DataIntegrityError) as exc_info:
        validate(
            dataset={"FIRST": _frame(["A B", "C D", "EFGH"], ["1", "nope", "bad"])},
            dictionary=_dictionary(["FIRST"]),
            feedback=False,
            fail_fast=True,
        )

    error = exc_info.value
    assert error.dataset.issues
    assert [
        (issue.type, issue.column, issue.ranges)
        for issue in error.dataset["FIRST"].issues
    ] == [(IssueType.PK_WHITESPACE, None, [Range(0, 1)])]
    _assert_first_check_error(error)
