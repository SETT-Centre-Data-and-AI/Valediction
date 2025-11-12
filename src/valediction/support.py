from dataclasses import dataclass
from datetime import datetime, timedelta
from math import trunc

BOLD_RED = "\033[1;31m"
BOLD_GREEN = "\033[1;92m"
RED = "\033[31m"
GREEN = "\033[92m"
RESET = "\033[0m"


@dataclass
class Runtime:
    message: str
    timedelta: timedelta


def print_bold_red(message: str, end: str | None = "\n") -> None:
    print(f"{BOLD_RED}{message}{RESET}", end=end)


def print_bold_green(message: str, end: str | None = "\n") -> None:
    print(f"{BOLD_GREEN}{message}{RESET}", end=end)


def print_green(message: str, end: str | None = "\n") -> None:
    print(f"{GREEN}{message}{RESET}", end=end)


def print_red(message: str, end: str | None = "\n") -> None:
    print(f"{RED}{message}{RESET}", end=end)


def list_as_bullets(elements: list, bullet: str = "\n - ") -> str:
    return bullet + bullet.join(elements)


def _normalise_name(name: str) -> str:
    return name.strip().upper()


def _get_runtime_string(runtime: timedelta) -> str:
    total_seconds = runtime.total_seconds()
    hours = trunc(total_seconds / 3600)
    minutes = trunc((total_seconds - (hours * 3600)) / 60)
    seconds = trunc((total_seconds - (hours * 3600) - (minutes * 60)) * 10) / 10
    runtime_string = (
        (f"{hours}h " if hours else "")
        + (f"{minutes}m " if minutes else "")
        + (f"{seconds}s" if not hours and not minutes else f"{trunc(seconds)}s")
    )
    return runtime_string


def calculate_runtime(start: datetime, stop: datetime | None = None) -> Runtime:
    """
    Summary:
        -   Takes two datetimes, and calculates the difference.
        -   Returns a message and raw timedelta as a named tuple, callable with .message or .delta

    Args:
        -   start (datetime):   Start time for calculation.
        -   stop (datetime):    Stop time for calculation. Defaults to now if not entered.

    Returns:
        tuple[str, timedelta]: Returns tuple, callable with .message (string) or .delta (raw timedelta)
    """
    stop = stop if stop else datetime.now()
    runtime = stop - start
    runtime_string = _get_runtime_string(runtime)

    return Runtime(message=runtime_string, timedelta=runtime)
