# valediction/io/csv_readers.py
from __future__ import annotations

import os
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterator, NamedTuple

import pandas as pd
from pandas import DataFrame
from pandas.errors import ParserError

from valediction.support import _normalise_name


class FrameChunk(NamedTuple):
    """A chunk of rows + I/O metadata.

    - start/end are 0-based inclusive row numbers in the full dataset.
    - file_pos/total_size/bytes_read are None when not reading from disk.
    """

    df: DataFrame
    start: int  # 0-based, inclusive
    end: int  # 10-based, inclusive
    total_size: int | None  # bytes of the whole file
    file_pos: int | None  # f.tell() after producing this chunk
    bytes_read: int | None  # bytes consumed to produce this chunk
    chunk_index: int | None  # 0-based index of this chunk

    # Cumulative Totals
    total_bytes_read: int | None
    total_chunks_seen: int | None

    def estimate_chunk_count(self) -> int:
        # Buffers (accounting for CSV tails/bytes innacuracy)
        EPS_ABS = 4096  # Fixed
        EPS_REL = 0.05  # 5% tail buffer

        bytes_seen = int(self.total_bytes_read)
        chunks_seen = max(1, int(self.total_chunks_seen))
        average = max(1.0, bytes_seen / float(chunks_seen))

        remaining = max(0, int(self.total_size) - bytes_seen)

        # Account for small tail if potentially complete
        tail_thresh = max(EPS_ABS, int(EPS_REL * average))
        if remaining <= tail_thresh:
            remaining = 0

        return chunks_seen + (0 if remaining == 0 else int(ceil(remaining / average)))

    def update_df(self, df: DataFrame) -> FrameChunk:
        return self._replace(df=df)


@dataclass(slots=True)
class CsvReadConfig:
    """Canonical CSV reading defaults for the overall project.

    Notes:
    - dtype="string" always reads columns as string, permitting downstream inference/validation.
    - keep_default_na=False and na_values=[] prevent pandas from coercing tokens like "NA".
    - We normalise headers and strip string values post-read (vectorised).
    """

    dtype: str = "string"
    keep_default_na: bool = False
    na_values: list[str] | None = None
    encoding: str = "utf-8"
    normalise_headers: bool = True
    strip_values: bool = True
    usecols: list[str] | None = None

    def __post_init__(self) -> None:
        if self.na_values is None:
            self.na_values = []


def _kwargs(cfg: CsvReadConfig | None = None) -> dict:
    cfg = cfg or CsvReadConfig()
    return dict(
        dtype=cfg.dtype,
        keep_default_na=cfg.keep_default_na,
        na_values=cfg.na_values,
        encoding=cfg.encoding,
        usecols=cfg.usecols,
    )


def _post_read_processing(df: DataFrame, cfg: CsvReadConfig) -> DataFrame:
    """Apply header normalisation and vectorised value stripping after reading."""
    cfg = cfg or CsvReadConfig()
    if cfg.normalise_headers:
        df = df.rename(columns={c: _normalise_name(c) for c in df.columns})
    if cfg.strip_values:
        str_cols = df.select_dtypes(include=["string"]).columns
        if len(str_cols) > 0:
            df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    return df


def read_csv_headers(path: str | Path, cfg: CsvReadConfig | None = None) -> DataFrame:
    """Read headers only (nrows=0) with canonical settings; returns a DataFrame."""
    cfg = cfg or CsvReadConfig()
    cfg.strip_values = False

    try:
        header = pd.read_csv(path, nrows=0, **_kwargs(cfg))
        return _post_read_processing(header, cfg)

    except ParserError as e:
        raise ParserError(
            f"Malformed CSV while reading header from '{path}': {e}"
        ) from e


def read_csv_all(path: str | Path, cfg: CsvReadConfig | None = None) -> FrameChunk:
    """Read the entire CSV with canonical settings; returns a single FrameChunk."""
    cfg = cfg or CsvReadConfig()
    try:
        file_size = os.path.getsize(path)

        with open(path, "rb") as file:
            start_pos = file.tell()
            df = pd.read_csv(file, **_kwargs(cfg))
            end_pos = file.tell()
        df = _post_read_processing(df, cfg)
        n = len(df)

        return FrameChunk(
            df=df,
            start=0,
            end=n - 1,
            total_size=file_size,
            file_pos=end_pos,
            bytes_read=end_pos - start_pos,
            chunk_index=1,
            total_bytes_read=file_size,
            total_chunks_seen=1,
        )
    except ParserError as e:
        raise ParserError(f"Malformed CSV while reading '{path}': {e}") from e


def read_csv_sample(
    path: str | Path, nrows: int, cfg: CsvReadConfig | None = None
) -> FrameChunk:
    """Read first `nrows` with canonical settings; returns a FrameChunk with I/O
    metadata."""
    cfg = cfg or CsvReadConfig()

    try:
        file_size = os.path.getsize(path)
        with open(path, "rb") as file:
            start_pos = file.tell()
            df = pd.read_csv(file, nrows=nrows, **_kwargs(cfg))
            end_pos = file.tell()

        df = _post_read_processing(df, cfg)
        n = len(df)

        bytes_read = (end_pos - start_pos) if end_pos > 0 else None
        file_pos = end_pos if end_pos > 0 else None

        return FrameChunk(
            df=df,
            start=0,
            end=n - 1,
            total_size=file_size,
            file_pos=file_pos,
            bytes_read=bytes_read,
            chunk_index=1,
            total_bytes_read=bytes_read or 0,
            total_chunks_seen=1,
        )

    except ParserError as e:
        raise ParserError(
            f"Malformed CSV while reading sample from '{path}': {e}"
        ) from e


def iter_csv_chunks(
    path: str | Path, chunk_size: int | None, cfg: CsvReadConfig | None = None
) -> Iterator[FrameChunk]:
    """Yield FrameChunk with canonical settings.

    Behaviour:
    - If chunk_size is None or <= 0: yields a single chunk for the entire file.
    - Else: yields multiple chunks each with populated bytes/position metadata.
    """
    cfg = cfg or CsvReadConfig()
    try:
        file_size = os.path.getsize(path)

        # No chunking: one full-file chunk with metadata
        if not chunk_size or (isinstance(chunk_size, int) and chunk_size <= 0):
            with open(path, "rb") as file:
                start_pos = file.tell()
                df = pd.read_csv(file, **_kwargs(cfg))
                end_pos = file.tell()
            df = _post_read_processing(df, cfg)
            n = len(df)
            if n == 0:
                return
            yield FrameChunk(
                df=df,
                start=0,
                end=n - 1,
                total_size=file_size,
                file_pos=end_pos,
                bytes_read=file_size,
                chunk_index=1,
                total_bytes_read=end_pos - start_pos,
                total_chunks_seen=1,
            )
            return

        # Chunking: stream with bytes/pos metadata
        with open(path, "rb") as file:
            reader = pd.read_csv(file, chunksize=chunk_size, **_kwargs(cfg))
            prev_pos = file.tell()
            offset = 0
            idx = 0
            cumulative_bytes = 0
            for raw in reader:
                idx += 1
                curr_pos = file.tell()
                bytes_read = max(0, curr_pos - prev_pos)
                prev_pos = curr_pos
                cumulative_bytes += bytes_read

                df = _post_read_processing(raw, cfg)
                n = len(df)
                if n == 0:
                    continue

                start = offset
                end = offset + n - 1
                offset += n

                yield FrameChunk(
                    df=df,
                    start=start,
                    end=end,
                    total_size=file_size,
                    file_pos=curr_pos,
                    bytes_read=bytes_read,
                    chunk_index=idx,
                    total_bytes_read=cumulative_bytes,
                    total_chunks_seen=idx,
                )

    except ParserError as e:
        raise ParserError(
            f"Malformed CSV while reading chunks from '{path}': {e}"
        ) from e


# Reading specific ranges
def _intersect_local_spans(
    ranges: list[tuple[int, int]],
    chunk_start: int,
    chunk_end: int,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r_start, r_end in ranges:
        lo = max(r_start, chunk_start)
        hi = min(r_end, chunk_end)
        if hi >= lo:
            out.append((lo - chunk_start, hi - chunk_start))
    return out


def read_csv_ranges(
    path: str | Path,
    ranges: list[tuple[int, int]],
    cfg: CsvReadConfig | None = None,
    chunk_size: int | None = 1_000_000,
) -> pd.DataFrame:
    """Read only the rows covered by `ranges` (global 0-based inclusive pairs).

    Respects CsvReadConfig (including usecols for column pruning). Returns a
    concatenated DataFrame (may be empty).
    """
    if not ranges:
        # honour columns if specified
        cols = cfg.usecols if (cfg and cfg.usecols) else None
        return pd.DataFrame(columns=cols) if cols else pd.DataFrame()

    parts: list[pd.DataFrame] = []
    for chunk in iter_csv_chunks(path, chunk_size=chunk_size, cfg=cfg):
        local_spans = _intersect_local_spans(ranges, chunk.start, chunk.end)
        if not local_spans:
            continue

        for lo, hi in local_spans:
            part = chunk.df.iloc[lo : hi + 1]
            parts.append(part)

    if not parts:
        cols = cfg.usecols if (cfg and cfg.usecols) else None
        return pd.DataFrame(columns=cols) if cols else pd.DataFrame()

    return pd.concat(parts, axis=0, ignore_index=False)
