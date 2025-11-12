from __future__ import annotations

import re
import warnings

import pandas as pd

from valediction.data_types.data_type_helpers import infer_datetime_format
from valediction.data_types.data_types import DataType
from valediction.integrity import get_config
from valediction.progress import Progress

# ---------- compiled patterns ----------
_INT_RE = re.compile(r"^[+-]?\d+$")
# FLOAT: allow decimals OR integers, plus optional scientific notation
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$")
# integers written as 123, 123.0, 123.
_INT_EQ_RE = re.compile(r"^[+-]?\d+(?:\.0*)?$")
_LEAD0_RE = re.compile(r"^[+-]?0\d+$")
_DATE_HINT_RE = re.compile(r"[-/T]")  # cheap prefilter
COLUMN_STEPS = 8


class ColumnState:
    def __init__(self, name: str) -> None:
        self.name = name
        self.data_type: DataType = DataType.TEXT
        self.nullable: bool = False
        self.max_length: int = 0

        # Locks / disqualifiers
        self.lock_text_due_to_leading_zero: bool = False
        self.lock_text_permanent: bool = False
        self.disqualify_numeric: bool = False
        self.disqualify_datetime: bool = False

        # Datetime speed hint
        self.cached_datetime_format: str | None = None
        self.prefer_date_first: bool = False

    def final_data_type_and_length(self) -> tuple[DataType, int | None]:
        def _len1() -> int:
            return max(1, self.max_length or 0)

        if self.lock_text_due_to_leading_zero or self.lock_text_permanent:
            return DataType.TEXT, _len1()
        if self.data_type == DataType.TEXT:
            return DataType.TEXT, _len1()

        if self.data_type == DataType.INTEGER:
            return DataType.INTEGER, None
        if self.data_type == DataType.FLOAT:
            return DataType.FLOAT, None
        if self.data_type == DataType.DATE:
            return DataType.DATE, None
        if self.data_type == DataType.DATETIME:
            return DataType.DATETIME, None

        return DataType.TEXT, _len1()


class TypeInferer:
    """
    Chunk-friendly type inference with:
      - compiled regex reuse
      - cached datetime formats
      - sticky TEXT on contradictions
      - unified debug logging via __say()
    """

    def __init__(
        self,
        *,
        dayfirst: bool,
        debug: bool = False,
        progress: Progress = None,
    ) -> None:
        config = get_config()
        self.dayfirst = dayfirst
        self.datetime_formats = config.date_formats
        self.null_tokens = {v.strip().lower() for v in config.null_values}
        self.states: dict[str, ColumnState] = {}
        self.debug = debug
        self.progress: Progress = progress
        self.__current_column: str | None = None

    # Inference
    def update_with_chunk(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        for col in df.columns:
            self.__current_column = col
            self.__begin_step(step="Preparing column")
            series = self._ensure_string_series(df[col])
            state = self.states.setdefault(col, ColumnState(name=col))
            self.__complete_step()  # 1 step

            trimmed, nulls, nonnull_mask, max_len = self._preprocess_column(
                series
            )  # 4 steps
            state.nullable |= bool(nulls.any())
            if max_len is not None and max_len > state.max_length:
                state.max_length = max_len

            if not bool(nonnull_mask.any()):
                self.__complete_step(n=3, save_as="Skipped")
                continue  # nothing to learn in this chunk

            non_nulls = trimmed[nonnull_mask]

            # Hard TEXT locks
            if self._apply_hard_text_locks(state, non_nulls):  # 1 step
                self.__complete_step(n=2, save_as="Skipped")
                continue

            # Datetime fast path
            if self._apply_datetime_fast_path(state, non_nulls):  # 1 step
                self.__complete_step(n=1, save_as="Skipped")
                continue

            # State-specific handling
            _handling_function: callable = {
                DataType.TEXT: self._handle_state_text,
                DataType.DATE: self._handle_state_date,
                DataType.DATETIME: self._handle_state_datetime,
                DataType.INTEGER: self._handle_state_integer,
                DataType.FLOAT: self._handle_state_float,
            }.get(state.data_type, self._handle_state_text)

            _handling_function(state, non_nulls)  # 1 of 5 steps

    # Inference Helpers
    @staticmethod
    def _ensure_string_series(s: pd.Series) -> pd.Series:
        if not pd.api.types.is_string_dtype(s.dtype):
            return s.astype("string")
        return s

    def _preprocess_column(
        self, s: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series, int | None]:
        self.__begin_step(step="Trimming whitespace")
        trimmed = s.str.strip()
        self.__complete_step()

        self.__begin_step(step="Checking nulls")
        nulls = trimmed.isna() | trimmed.str.lower().isin(self.null_tokens)
        self.__complete_step()

        self.__begin_step(step="Checking max length")
        lengths = s.str.len()
        max_len = int(lengths.max(skipna=True)) if lengths.notna().any() else None
        self.__complete_step()

        self.__begin_step(step="Setting non-null mask")
        nonnull_mask = (~nulls) & s.notna()
        self.__complete_step()

        return trimmed, nulls, nonnull_mask, max_len

    # Early Locks
    @staticmethod
    def _looks_dateish(nn: pd.Series) -> bool:
        return bool(nn.str.contains(_DATE_HINT_RE).any())

    @staticmethod
    def _has_leading_zero(nn: pd.Series) -> bool:
        return bool(nn.str.match(_LEAD0_RE, na=False).any())

    def _apply_hard_text_locks(self, st: ColumnState, nn: pd.Series) -> bool:
        if st.lock_text_due_to_leading_zero or st.lock_text_permanent:
            self._transition(st, DataType.TEXT, "locked to TEXT")
            self.__complete_step()
            return True

        if self._has_leading_zero(nn):
            self._debug_leading_zero_examples(st, nn)
            st.lock_text_due_to_leading_zero = True
            self._transition(st, DataType.TEXT, "leading-zero integer tokens")
            self.__complete_step()
            return True

        self.__complete_step()
        return False

    def _apply_datetime_fast_path(self, st: ColumnState, nn: pd.Series) -> bool:
        self.__begin_step(step="Applying datetime locks")

        # Cached single format
        if st.cached_datetime_format is not None:
            ok, has_time = self._parse_with_cached_format(nn, st.cached_datetime_format)
            if ok.all():
                self._transition(
                    st,
                    DataType.DATETIME if has_time.any() else DataType.DATE,
                    f"cached datetime format={st.cached_datetime_format!r}",
                )
                self.__complete_step()
                return True

            st.cached_datetime_format = None
            st.prefer_date_first = False

        # Date-first hint (explicit formats)
        if st.prefer_date_first and not st.disqualify_datetime:
            for fmt in self.datetime_formats:
                ok, has_time = self._parse_with_cached_format(nn, fmt)
                if ok.all():
                    st.cached_datetime_format = fmt
                    self._transition(
                        st,
                        DataType.DATETIME if has_time.any() else DataType.DATE,
                        f"explicit datetime format={fmt!r}",
                    )
                    self.__complete_step()
                    return True

        self.__complete_step()
        return False

    # State Handlers
    def _handle_state_text(self, st: ColumnState, nn: pd.Series) -> None:
        self.__begin_step(step="Handling text")
        # DATETIME attempt
        if not st.disqualify_datetime and self._looks_dateish(nn):
            if self._try_parse_datetime_then_cache(st, nn):
                self.__complete_step()
                return

        # NUMERIC attempt
        if not st.disqualify_numeric:
            int_equiv = nn.str.fullmatch(_INT_EQ_RE, na=False)
            float_like = nn.str.fullmatch(_FLOAT_RE, na=False)

            if int_equiv.all():
                self._transition(st, DataType.INTEGER, "all integer-equivalent")
                self.__complete_step()
                return

            if (int_equiv | float_like).all():
                self._debug_float_promotion(st, nn, int_equiv, float_like)
                self._transition(st, DataType.FLOAT, "mixed numeric (int/float)")
                self.__complete_step()
                return

            # Otherwise: non-numeric → TEXT (sticky)
            self._debug_offenders_numeric(st, nn, int_equiv, float_like)
            st.disqualify_numeric = True
            st.lock_text_permanent = True
            self._transition(st, DataType.TEXT, "non-numeric tokens present")
            self.__complete_step()
            return

        # If both numeric and datetime are disqualified, permanently TEXT
        if st.disqualify_numeric and st.disqualify_datetime:
            st.lock_text_permanent = True
            self._transition(
                st, DataType.TEXT, "both numeric and datetime disqualified"
            )
        self.__complete_step()

    def _handle_state_date(self, st: ColumnState, nn: pd.Series) -> None:
        self.__begin_step(step="Handling dates")
        if not self._looks_dateish(nn):
            st.disqualify_datetime = True
            st.lock_text_permanent = True
            self._transition(st, DataType.TEXT, "lost date-ish pattern")
            self.__complete_step()
            return

        ok, has_time = self._datetime_parse_ok(nn)
        if not ok.all():
            self._debug_offenders_datetime(st, nn, ok)
            st.disqualify_datetime = True
            st.lock_text_permanent = True
            self._transition(st, DataType.TEXT, "datetime parse failures")
        elif has_time.any():
            self._transition(st, DataType.DATETIME, "time component detected")

        self.__complete_step()

    def _handle_state_datetime(self, st: ColumnState, nn: pd.Series) -> None:
        self.__begin_step(step="Handling datetimes")
        if not self._looks_dateish(nn):
            st.disqualify_datetime = True
            st.lock_text_permanent = True
            self._transition(st, DataType.TEXT, "lost date-ish pattern")
            self.__complete_step()
            return

        ok, _ = self._datetime_parse_ok(nn)
        if not ok.all():
            self._debug_offenders_datetime(st, nn, ok)
            st.disqualify_datetime = True
            st.lock_text_permanent = True
            self._transition(st, DataType.TEXT, "datetime parse failures")

        self.__complete_step()

    def _handle_state_integer(self, st: ColumnState, nn: pd.Series) -> None:
        self.__begin_step(step="Handling integers")
        int_equiv = nn.str.fullmatch(_INT_EQ_RE, na=False)
        float_like = nn.str.fullmatch(_FLOAT_RE, na=False)

        if not (int_equiv | float_like).all():
            self._debug_offenders_numeric(st, nn, int_equiv, float_like)
            st.disqualify_numeric = True
            st.lock_text_permanent = True
            self._transition(st, DataType.TEXT, "non-numeric tokens introduced")
        elif float_like.any() and not int_equiv.all():
            self._debug_float_promotion(st, nn, int_equiv, float_like)
            self._transition(st, DataType.FLOAT, "decimals/scientific detected")

        self.__complete_step()
        # else remain INTEGER

    def _handle_state_float(self, st: ColumnState, nn: pd.Series) -> None:
        self.__begin_step(step="Handling floats")
        int_like = nn.str.fullmatch(_INT_RE, na=False)
        fl_like = nn.str.fullmatch(_FLOAT_RE, na=False)
        if not (int_like | fl_like).all():
            self._debug_offenders_numeric(st, nn, int_like, fl_like)
            st.disqualify_numeric = True
            st.lock_text_permanent = True
            self._transition(st, DataType.TEXT, "non-numeric tokens introduced")
        self.__complete_step()

    # Datetime Parsing
    def _try_parse_datetime_then_cache(self, st: ColumnState, nn: pd.Series) -> bool:
        # 1) If we’ve already cached a format, try it fast
        if st.cached_datetime_format is not None:
            ok, has_time = self._parse_with_cached_format(nn, st.cached_datetime_format)
            if ok.all():
                self._transition(
                    st,
                    DataType.DATETIME if has_time.any() else DataType.DATE,
                    f"cached datetime format={st.cached_datetime_format!r}",
                )
                return True
            # cache failed on this chunk; clear and fall through to re-infer once
            st.cached_datetime_format = None
            st.prefer_date_first = False

        # 2) Infer with the new helper (efficient: unique, batched, intersects across slices)
        #    Work on uniques only for speed and stability.
        uniq = (
            nn.astype("string", copy=False)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .unique()
        )
        if len(uniq) == 0:
            return False

        try:
            fmt_or_false = infer_datetime_format(pd.Series(uniq, dtype="string"))
        except ValueError as e:
            # ambiguous after scanning – treat as “can’t determine” and disqualify
            self.__say(f"[{st.name}] datetime ambiguous: {e}")
            st.disqualify_datetime = True
            return False

        if fmt_or_false is False:
            # helper couldn’t find any valid explicit format
            st.disqualify_datetime = True
            self._transition(
                st, DataType.TEXT, "datetime helper found no matching format"
            )
            return False

        # 3) Cache and confirm on current (non-unique) values
        st.cached_datetime_format = fmt_or_false
        st.prefer_date_first = True
        ok, has_time = self._parse_with_cached_format(nn, st.cached_datetime_format)
        if ok.all():
            self._transition(
                st,
                DataType.DATETIME if has_time.any() else DataType.DATE,
                f"explicit datetime format={st.cached_datetime_format!r}",
            )
            return True

        self.__say(
            f"[{st.name}] cached format failed on live slice; disqualifying datetime."
        )
        st.cached_datetime_format = None
        st.disqualify_datetime = True
        return False

    def _parse_with_cached_format(
        self, s: pd.Series, fmt: str
    ) -> tuple[pd.Series, pd.Series]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(s, format=fmt, errors="coerce", utc=False)

        ok = parsed.notna()
        has_time = ok & (
            (parsed.dt.hour != 0)
            | (parsed.dt.minute != 0)
            | (parsed.dt.second != 0)
            | (parsed.dt.microsecond != 0)
        )
        return ok, has_time

    def _datetime_parse_ok(self, s: pd.Series) -> tuple[pd.Series, pd.Series]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(
                s, errors="coerce", dayfirst=self.dayfirst, utc=False
            )

        ok = parsed.notna()
        has_time = ok & (
            (parsed.dt.hour != 0)
            | (parsed.dt.minute != 0)
            | (parsed.dt.second != 0)
            | (parsed.dt.microsecond != 0)
        )
        return ok, has_time

    # Debug/Log
    def __say(self, *values: object, sep: str = " ", end: str = "\n") -> None:
        if self.debug:
            print("TypeInferer:", *values, sep=sep, end=end)

    def _transition(self, st: ColumnState, to_type: DataType, reason: str) -> None:
        """Set st.data_type and emit a standardised debug line if changed."""
        from_type = st.data_type
        st.data_type = to_type
        if self.debug:
            if from_type != to_type:
                self.__say(f"[{st.name}] {from_type.name} → {to_type.name} ({reason})")
            else:
                self.__say(f"[{st.name}] stays {to_type.name} ({reason})")

    def _fmt_examples(
        self,
        vc: pd.Series,
        *,
        max_examples: int = 5,
        max_value_len: int = 80,
    ) -> str:
        shown = vc.head(max_examples)
        parts: list[str] = []
        for val in shown.index:
            s = repr(val)
            if len(s) > max_value_len:
                s = s[: max_value_len - 1] + "…"
            parts.append(s)
        extra = vc.shape[0] - shown.shape[0]
        suffix = f"; …+{extra}" if extra > 0 else ""
        return "[" + "; ".join(parts) + suffix + "]"

    def _debug_offenders_numeric(
        self,
        st: ColumnState,
        nn: pd.Series,
        int_like: pd.Series,
        float_like: pd.Series,
        *,
        max_examples: int = 5,
        note: str = "non-numeric present",
    ) -> None:
        if not self.debug:
            return
        bad = ~(int_like | float_like)
        if not bool(bad.any()):
            return
        vc = nn[bad].value_counts(dropna=False)
        examples = self._fmt_examples(vc, max_examples=max_examples)
        self.__say(f"[{st.name}] numeric disqualified: {note}. Examples {examples}")

    def _debug_offenders_datetime(
        self,
        st: ColumnState,
        nn: pd.Series,
        ok_mask: pd.Series,
        *,
        max_examples: int = 5,
    ) -> None:
        if not self.debug:
            return
        bad = ~ok_mask
        if not bool(bad.any()):
            return
        vc = nn[bad].value_counts(dropna=False)
        examples = self._fmt_examples(vc, max_examples=max_examples)
        self.__say(f"[{st.name}] datetime disqualified. Examples {examples}")

    def _debug_leading_zero_examples(
        self,
        st: ColumnState,
        nn: pd.Series,
        *,
        max_examples: int = 5,
    ) -> None:
        if not self.debug:
            return
        m = nn.str.match(_LEAD0_RE, na=False)
        if not bool(m.any()):
            return
        vc = nn[m].value_counts(dropna=False)
        examples = self._fmt_examples(vc, max_examples=max_examples)
        self.__say(f"[{st.name}] leading-zero lock. Examples {examples}")

    def _debug_float_promotion(
        self,
        st: ColumnState,
        nn: pd.Series,
        int_equiv: pd.Series,
        float_like: pd.Series,
        *,
        max_examples: int = 5,
    ) -> None:
        if not self.debug:
            return
        non_integer_numeric = float_like & ~int_equiv
        if not bool(non_integer_numeric.any()):
            self.__say(f"[{st.name}] promoted to FLOAT.")
            return
        sample = nn[non_integer_numeric]
        reasons = []
        if bool(sample.str.contains(r"\.", na=False).any()):
            reasons.append("decimals present")
        if bool(sample.str.contains(r"[eE][+-]?\d+", na=False).any()):
            reasons.append("scientific notation present")
        reason_msg = (": " + ", ".join(reasons)) if reasons else ""
        vc = sample.value_counts(dropna=False)
        examples = self._fmt_examples(vc, max_examples=max_examples)
        self.__say(f"[{st.name}] promoted to FLOAT{reason_msg}. Examples {examples}")

    def __begin_step(self, step: str):
        self.progress.begin_step(
            step=step, alt_postfix=f"{self.__current_column}: {step}"
        )

    def __complete_step(self, n: int = 1, save_as: str = None):
        self.progress.complete_step(n=n, save_as=save_as)
