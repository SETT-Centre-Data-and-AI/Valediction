from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pandas import DataFrame

DataLike = Path | str | DataFrame


@runtime_checkable
class DatasetItemLike(Protocol):
    name: str
    data: DataLike
    _dictionary_runtimes: dict[str, timedelta]
    _padding: int

    @property
    def is_path(self) -> bool: ...
    @property
    def is_dataframe(self) -> bool: ...
    @property
    def data(self) -> Any: ...
    @property
    def column_count(self) -> int: ...
    @property
    def primary_keys(self) -> list[str]: ...


def as_folder(path_like: str | Path) -> Path:
    """Coerce to a Path and verify it is an existing directory."""
    p = Path(path_like)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Folder not found or not a directory: {p}")
    return p


def list_csvs(folder: str | Path, *, recursive: bool = False) -> list[Path]:
    """Return CSV files in a folder.

    Non-recursive by default.
    """
    p = as_folder(folder)
    pattern = "**/*.csv" if recursive else "*.csv"
    return sorted(p.glob(pattern))
