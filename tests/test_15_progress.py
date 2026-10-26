from time import time

from valediction.progress import (
    FORMAT_KNOWN_TOTAL,
    FORMAT_UNKNOWN_TOTAL,
    Progress,
    _AlignedTqdm,
    _format_rate,
)


def test_rate_format_stays_aligned_for_one_to_three_digit_rates() -> None:
    rates = [1.23, 12.34, 123.45]
    formatted = [_format_rate(rate, "step") for rate in rates]

    assert [value.index(".") for value in formatted] == [3, 3, 3]
    assert [value.index("step/s") for value in formatted] == [6, 6, 6]
    assert _format_rate(0.5, "step") == "  2.00s/step"


def test_completed_bar_falls_back_to_overall_rate() -> None:
    bar = _AlignedTqdm(total=1, unit="step", disable=True)
    try:
        bar.n = 1
        bar.start_t = time() - 2
        bar._ema_dt = lambda: 0

        assert "?step/s" not in str(bar)
    finally:
        bar.close()


def test_unknown_total_displays_placeholder_with_aligned_padding() -> None:
    bar = _AlignedTqdm(total=None, unit="step", disable=True)
    try:
        rendered = bar.format_meter(
            **bar.format_dict,
            bar_format=FORMAT_UNKNOWN_TOTAL,
        )
        assert "|   -/?   [" in rendered
        assert "00:00<??:??," in rendered
    finally:
        bar.close()


def test_unknown_total_initial_render_uses_aligned_format(
    capsys,
) -> None:
    progress = Progress(desc="Validating lab_tests:", est_total=None)
    try:
        progress.begin_step("Checking vocabularies")
    finally:
        progress.close()

    output = capsys.readouterr().err
    assert "0step [" not in output
    assert "Validating lab_tests:   0%|" in output
    assert "-/?" in output


def test_known_total_step_count_uses_three_character_fields() -> None:
    bar = _AlignedTqdm(total=15, unit="step", disable=True)
    try:
        for completed, total, expected in (
            (15, 15, "  15/15  "),
            (3, 3, "   3/3   "),
        ):
            bar.n = completed
            bar.total = total
            format_dict = bar.format_dict
            format_dict["bar_format"] = FORMAT_KNOWN_TOTAL
            rendered = bar.format_meter(**format_dict)

            assert expected in rendered
    finally:
        bar.close()


def test_finish_completes_bar_by_default() -> None:
    progress = Progress(est_total=5)
    try:
        progress.complete_step(n=2)

        progress.finish()

        assert progress.bar.total == 2
        assert progress.bar.n == 2
        assert progress.completed_steps == 2
    finally:
        progress.close()


def test_finish_completes_empty_bar_by_default() -> None:
    progress = Progress(est_total=5)
    try:
        progress.finish()

        assert progress.bar.total == 1
        assert progress.bar.n == 1
        assert progress.completed_steps == 1
    finally:
        progress.close()


def test_incomplete_finish_preserves_progress() -> None:
    progress = Progress(est_total=5)
    try:
        progress.complete_step(n=2)

        progress.finish(
            postfix="Failed fast with issues",
            good=False,
            complete=False,
        )

        assert progress.bar.total == 5
        assert progress.bar.n == 2
        assert progress.completed_steps == 2
        assert "Failed fast with issues" in progress.postfix
    finally:
        progress.close()
