from valediction.progress import Progress


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
