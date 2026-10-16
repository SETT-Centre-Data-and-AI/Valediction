import csv
import gc
import os
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import numpy as np
import pytest
from pandas import DataFrame, date_range

from valediction.datasets.datasets import Dataset, DatasetItem
from valediction.dictionary.model import Column, Dictionary, Table
from valediction.validation.validation import Validator

TABLE_NAME = "PERFORMANCE"
ROW_COUNT = 1_000_000
RUN_COUNT = 10
EXPECTED_REPORTED_STEPS = {
    "Importing data",
    "Checking for missing columns",
    "Checking for extra columns",
    "Setting nulls",
    "Checking for column nulls",
    "Checking for primary key whitespace",
    "Checking primary key text lengths",
    "Creating primary key hashes",
    "Checking for primary key nulls",
    "Checking for primary key collision",
    "Caching primary keys",
    "Inferring datetime formats",
    "Checking column types",
    "Checking text lengths",
    "Checking for forbidden characters",
    "apply_dictionary",
    "Total validation",
}


def _repeat_values(values: list[str], size: int) -> list[str]:
    return [values[index % len(values)] for index in range(size)]


def _make_dataframe() -> DataFrame:
    text_values = [f"value-{index:03d}" for index in range(100)]
    dates = (
        date_range("2020-01-01", periods=365, freq="D").strftime("%Y-%m-%d").tolist()
    )
    timestamps = (
        date_range("2020-01-01", periods=31 * 24, freq="h")
        .strftime("%Y-%m-%d %H:%M:%S")
        .tolist()
    )
    date_values = _repeat_values(dates, ROW_COUNT)
    timestamp_values = _repeat_values(timestamps, ROW_COUNT)

    return DataFrame(
        {
            "INT": np.arange(ROW_COUNT, dtype=np.int64),
            "STR": _repeat_values(text_values, ROW_COUNT),
            "FLOAT": np.arange(ROW_COUNT, dtype=np.float64) / 10,
            "DATE": date_values,
            "TIMESTAMP": timestamp_values,
        }
    )


def _make_dictionary() -> Dictionary:
    return Dictionary(
        name="Performance Test",
        tables=[
            Table(
                name=TABLE_NAME,
                columns=[
                    Column(name="INT", order=1, data_type="int", primary_key=1),
                    Column(name="STR", order=2, data_type="str", length=9),
                    Column(name="FLOAT", order=3, data_type="float"),
                    Column(name="DATE", order=4, data_type="date"),
                    Column(name="TIMESTAMP", order=5, data_type="timestamp"),
                ],
            )
        ],
    )


def _write_timing_report(
    report_path: Path,
    timing_rows: list[tuple[int | str, str, int | str, float | str]],
    mean_rows: list[tuple[int | str, str, int | str, float | str]],
    message: str,
) -> None:
    with report_path.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.writer(report_file)
        writer.writerow(["run", "step", "occurrence", "duration_seconds", "message"])
        writer.writerows((*row, message) for row in timing_rows)
        writer.writerows((*row, message) for row in mean_rows)


def _ensure_history_message_column(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return

    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source))
    if "message" in rows[0]:
        return

    rows[0].append("message")
    for row in rows[1:]:
        row.append("")

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            csv.writer(temporary_file).writerows(rows)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def test_timing_report_records_message(tmp_path: Path) -> None:
    report_path = tmp_path / "timings.csv"
    message = "Testing validation changes before commit"
    _write_timing_report(
        report_path,
        [(1, "Checking column types", 1, 0.25)],
        [("MEAN", "Checking column types", "", 0.25)],
        message,
    )

    with report_path.open(newline="", encoding="utf-8") as report_file:
        rows = list(csv.DictReader(report_file))

    assert [row["message"] for row in rows] == [message, message]


def test_history_schema_upgrade_preserves_existing_rows(tmp_path: Path) -> None:
    history_path = tmp_path / "history.csv"
    history_path.write_text("git_hash,mean_seconds\nabc,1.25\n", encoding="utf-8")

    _ensure_history_message_column(history_path)

    with history_path.open(newline="", encoding="utf-8") as history_file:
        rows = list(csv.DictReader(history_file))
    assert rows == [{"git_hash": "abc", "mean_seconds": "1.25", "message": ""}]


def test_validation_timings(monkeypatch) -> None:
    raw_data = _make_dataframe()
    timing_rows: list[tuple[int | str, str, int | str, float]] = []
    active_steps: dict[int, tuple[str, float]] = {}
    occurrence_counts: dict[tuple[int, str], int] = defaultdict(int)
    current_run = 0

    original_begin_step = Validator._Validator__begin_step
    original_complete_step = Validator._Validator__complete_step
    original_apply_dictionary = DatasetItem.apply_dictionary

    def timed_begin_step(validator: Validator, step: str | None = None) -> None:
        original_begin_step(validator, step=step)
        active_steps[id(validator)] = (step or "Unnamed step", perf_counter())

    def timed_complete_step(validator: Validator) -> None:
        active = active_steps.pop(id(validator), None)
        if active is not None:
            step, started = active
            occurrence_counts[(current_run, step)] += 1
            timing_rows.append(
                (
                    current_run,
                    step,
                    occurrence_counts[(current_run, step)],
                    perf_counter() - started,
                )
            )
        original_complete_step(validator)

    def timed_apply_dictionary(item: DatasetItem) -> None:
        started = perf_counter()
        original_apply_dictionary(item)
        timing_rows.append(
            (current_run, "apply_dictionary", 1, perf_counter() - started)
        )

    monkeypatch.setattr(Validator, "_Validator__begin_step", timed_begin_step)
    monkeypatch.setattr(Validator, "_Validator__complete_step", timed_complete_step)
    monkeypatch.setattr(DatasetItem, "apply_dictionary", timed_apply_dictionary)

    expected_dtypes = {
        "INT": "Int64",
        "STR": "string",
        "FLOAT": "Float64",
        "DATE": "datetime64[ns]",
        "TIMESTAMP": "datetime64[ns]",
    }

    for current_run in range(1, RUN_COUNT + 1):
        frame = raw_data.copy(deep=True)
        dataset = Dataset.create_from({TABLE_NAME: frame})
        dataset.import_dictionary(_make_dictionary())

        started = perf_counter()
        dataset.validate(feedback=False)
        timing_rows.append(
            (
                current_run,
                "Total validation",
                1,
                perf_counter() - started,
            )
        )

        assert dataset.check()
        validated_data = dataset[TABLE_NAME].data
        assert {
            column: str(dtype) for column, dtype in validated_data.dtypes.items()
        } == (expected_dtypes)
        assert validated_data["DATE"].iloc[1].hour == 0
        assert validated_data["TIMESTAMP"].iloc[1].hour == 1

        del dataset, frame, validated_data
        gc.collect()

    repository_root = Path(__file__).resolve().parents[1]
    report_directory = Path(__file__).parent / "reports"
    report_directory.mkdir(parents=True, exist_ok=True)
    performance_message = os.environ.get("VALEDICTION_PERFORMANCE_MESSAGE", "").strip()

    durations_by_step: dict[str, list[float]] = defaultdict(list)
    for _, step, _, duration in timing_rows:
        durations_by_step[step].append(duration)
    assert EXPECTED_REPORTED_STEPS <= durations_by_step.keys()
    mean_rows = [
        ("MEAN", step, "", mean(durations))
        for step, durations in durations_by_step.items()
    ]
    mean_validation_seconds = mean(durations_by_step["Total validation"])
    report_name = f"{datetime.now():%Y%m%d %H%M%S} {mean_validation_seconds:.3f}s.csv"
    report_path = report_directory / report_name

    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    recorded_at = datetime.now().isoformat(timespec="seconds")
    total_history_path = report_directory / "performance_history.csv"
    full_history_path = report_directory / "performance_history_full.csv"
    _ensure_history_message_column(total_history_path)
    _ensure_history_message_column(full_history_path)

    total_history_exists = (
        total_history_path.exists() and total_history_path.stat().st_size > 0
    )
    with total_history_path.open("a", newline="", encoding="utf-8") as history_file:
        writer = csv.writer(history_file)
        if not total_history_exists:
            writer.writerow(
                ["git_hash", "recorded_at", "mean_total_validation_seconds", "message"]
            )
        writer.writerow(
            (
                git_hash,
                recorded_at,
                f"{mean(durations_by_step['Total validation']):.6f}",
                performance_message,
            )
        )

    full_history_exists = (
        full_history_path.exists() and full_history_path.stat().st_size > 0
    )
    with full_history_path.open("a", newline="", encoding="utf-8") as history_file:
        writer = csv.writer(history_file)
        if not full_history_exists:
            writer.writerow(
                ["git_hash", "recorded_at", "step", "mean_seconds", "message"]
            )
        writer.writerows(
            (git_hash, recorded_at, step, f"{mean(durations):.6f}", performance_message)
            for step, durations in durations_by_step.items()
        )

    formatted_timing_rows = [
        (run, step, occurrence, f"{duration:.3f}")
        for run, step, occurrence, duration in timing_rows
    ]
    formatted_mean_rows = [
        (run, step, occurrence, f"{duration:.3f}")
        for run, step, occurrence, duration in mean_rows
    ]
    _write_timing_report(
        report_path,
        formatted_timing_rows,
        formatted_mean_rows,
        performance_message,
    )

    print("Validation timing summary (seconds)")
    print(f"Performance message: {performance_message or '(none)'}")
    for step, durations in durations_by_step.items():
        print(
            f"{step}: mean={mean(durations):.3f}, "
            f"median={median(durations):.3f}, runs={len(durations)}"
        )
    print(f"Detailed timing report: {report_path}")
    print(f"Total-time history: {total_history_path}")
    print(f"Full step history: {full_history_path}")

    assert (
        len([row for row in timing_rows if row[1] == "Total validation"]) == RUN_COUNT
    )
    assert (
        len([row for row in timing_rows if row[1] == "apply_dictionary"]) == RUN_COUNT
    )
    assert len(timing_rows) > RUN_COUNT
    assert len(mean_rows) == len(durations_by_step)
    assert not active_steps


if __name__ == "__main__":
    pytest.main([__file__])
