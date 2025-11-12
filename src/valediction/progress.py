# progress.py
from __future__ import annotations

from datetime import datetime, timedelta

from tqdm import tqdm

from valediction.support import BOLD_GREEN, BOLD_RED, RESET, calculate_runtime

FORMAT_KNOWN_TOTAL = (
    "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)

FORMAT_UNKNOWN_TOTAL = (
    "{desc} {percentage:3.0f}%|{bar}| ?/? [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)


class Progress:
    def __init__(
        self,
        desc: str = "",
        est_total: int | None = 1,
        smoothing_steps: int = 0,
        unit: str = "step",
        starting_step: str | None = None,
        enabled: bool = True,
    ) -> None:
        """Progress bar (tqdm) with manual control.

        Args:
            desc (str): label shown to the left of the bar
            starting_step (str, optional): initial step and starting postfix, e.g. "Importing Data".
                Defaults to "".
            est_total (int, optional): initial total number of steps (can grow/shrink later).
                Defaults to 1.
            smoothing_steps (int, optional): window length of previous steps to approximate ETA.
                Use 0 for global average. Defaults to 0.
            unit (str, optional): display unit (default: "step"). Defaults to "step".
            bar_format (str, optional): custom bar format. Defaults to None (using Progress
                default).
            enabled (bool, optional): Enables switching off, avoiding duplication of upstream
            checks. Defaults to True.
        """
        self.enabled: bool = enabled
        self.desc: str = desc
        self.est_total: int = est_total
        self.smoothing_steps: int = max(0, int(smoothing_steps or 0))
        self.unit: str = unit
        self.postfix: str = ""

        # Bar
        self.bar: tqdm = None
        self.total_steps: int = self.est_total
        self.completed_steps: int = 0

        # Runtimes
        self.full_start: datetime = None
        self.step_start: datetime = None
        self.current_step = starting_step or ""
        self.runtimes: dict[str, timedelta] = {}

        self.__init_progress_bar()

    # Context
    def __enter__(self) -> Progress:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # Initialisation
    def __init_progress_bar(self) -> None:
        now = datetime.now()
        self.full_start = now
        self.step_start = now

        if not self.enabled:
            return

        smoothing = (
            0.0 if self.smoothing_steps == 0 else 2.0 / (self.smoothing_steps + 1)
        )

        self.bar = tqdm(
            total=self.total_steps,
            unit=self.unit,
            desc=self.desc,
            smoothing=smoothing,
        )
        self.__set_bar_format()
        if self.current_step:
            self.bar.set_postfix_str(self.current_step)

    def __set_bar_format(self) -> None:
        if self.est_total:
            self.bar.bar_format = FORMAT_KNOWN_TOTAL
        else:
            self.bar.bar_format = FORMAT_UNKNOWN_TOTAL

    # Management
    def retarget_total(self, new_total: int) -> None:
        if not self.enabled:
            return

        new_total = max(1, int(new_total))
        self.total_steps = new_total
        self.est_total = new_total
        self.__set_bar_format()

        if self.bar is None:
            return

        if int(self.bar.total or 0) == new_total:
            return

        self.bar.total = new_total
        self._refresh()

    def begin_step(self, step: str, alt_postfix: str = None) -> None:
        self.step_start = datetime.now()
        self.current_step = step
        postfix = alt_postfix or self.current_step

        if self.enabled:
            self._set_postfix(postfix)
            self._refresh()

    def complete_step(
        self, n: int = 1, from_time: datetime = None, save_as: str = None
    ) -> None:
        step = save_as or self.current_step
        runtime = calculate_runtime(start=from_time or self.step_start)
        if self.runtimes.get(step) is None:
            self.runtimes[step] = runtime.timedelta
        else:
            self.runtimes[step] += runtime.timedelta

        if self.enabled:
            self._tick(n=n)

    def finish(
        self,
        postfix: str | None = "Completed",
        save_as: str = "Total",
        good: bool = None,
    ) -> None:
        self.complete_step(n=0, from_time=self.full_start, save_as=save_as)

        if not self.enabled:
            return

        postfix = (
            f"{BOLD_GREEN if good else BOLD_RED if good is False else ''}"
            + postfix
            + f"{'' if good is None else RESET}"
        )
        self._set_postfix(postfix)
        completed_steps = int(getattr(self.bar, "n", 0))
        if completed_steps <= 0:
            self.bar.total = 1
            self.bar.update(1)
            self.completed_steps = 1

        else:
            self.bar.total = completed_steps
            if self.bar.n < completed_steps:
                self.bar.update(completed_steps - self.bar.n)
            self.completed_steps = completed_steps
            self._refresh()

    def close(self) -> None:
        if not self.enabled:
            return

        if self.bar:
            try:
                self.bar.close()
            finally:
                self.bar = None

    # Helpers
    def _refresh(self) -> None:
        if not self.enabled:
            return

        self.bar.refresh()

    def _tick(self, n: int = 1):
        self.completed_steps += n
        if not self.enabled:
            return

        if n:
            self.bar.update(n)
            self._refresh()

    def _set_postfix(self, postfix: str) -> None:
        if not self.enabled:
            return

        postfix = postfix or ""
        self.postfix = postfix
        self.bar.set_postfix_str(postfix)
        self._refresh()
