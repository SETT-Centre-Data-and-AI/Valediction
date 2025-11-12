from contextlib import contextmanager
from pathlib import Path

import pytest
from support import compare_dictionary_columns, compare_dictionary_tables

from valediction.demo.demo_dictionary import demo_dictionary
from valediction.dictionary.importing import import_dictionary

EXPORT_DIR = Path(__file__).parent
EXPORT_FILENAME = "TEST - Data Dictionary"


@contextmanager
def cleanup():
    path = EXPORT_DIR / f"{EXPORT_FILENAME}.xlsx"
    if path.exists():
        path.unlink()
    try:
        yield path  # caller uses this path
    finally:
        try:
            path.unlink()  # post-clean
        except FileNotFoundError:
            pass


def test_dictionary_export() -> None:
    with cleanup():
        # Export Data Dictionary
        dictionary = demo_dictionary()
        dictionary.export_dictionary(directory=EXPORT_DIR, filename=EXPORT_FILENAME)


def test_dictionary_export_format() -> None:
    with cleanup():
        # Export Demo Data Dictionary
        dictionary = demo_dictionary()
        dictionary.export_dictionary(directory=EXPORT_DIR, filename=EXPORT_FILENAME)

        # Reimport Demo Data Dictionary
        exported = import_dictionary(EXPORT_DIR / f"{EXPORT_FILENAME}.xlsx")

        # Check Match
        compare_dictionary_tables(exported, dictionary)
        compare_dictionary_columns(exported, dictionary)


def test_dictionary_export_without_metadata() -> None:
    with cleanup():
        # Export Data Dictionary
        dictionary = demo_dictionary()
        dictionary.name = None
        dictionary.version = None
        dictionary.inclusion_criteria = None
        dictionary.exclusion_criteria = None
        dictionary.version_notes = None
        dictionary.export_dictionary(
            directory=EXPORT_DIR, filename=EXPORT_FILENAME, overwrite=True
        )

        # Reimport Demo Data Dictionary
        exported = import_dictionary(EXPORT_DIR / f"{EXPORT_FILENAME}.xlsx")

        # Check Match
        compare_dictionary_tables(exported, dictionary)
        compare_dictionary_columns(exported, dictionary)


if __name__ == "__main__":
    pytest.main([__file__])
