import pytest
from support import compare_dictionary_columns, compare_dictionary_tables

from valediction.demo import DEMO_DICTIONARY
from valediction.demo.demo_dictionary import demo_dictionary
from valediction.dictionary.importing import import_dictionary
from valediction.dictionary.model import Dictionary


def check_dictionary() -> Dictionary:
    return demo_dictionary()


def imported_dictionary() -> Dictionary:
    return import_dictionary(DEMO_DICTIONARY)


def test_demo_dictionary_import():
    imported_dictionary = import_dictionary(DEMO_DICTIONARY)
    assert imported_dictionary


def test_tables_imported_correctly():
    test_dictionary = imported_dictionary()
    check_dictionary = demo_dictionary()

    compare_dictionary_tables(test_dictionary, check_dictionary)


def test_columns_imported_correctly():
    test_dictionary = imported_dictionary()
    check_dictionary = demo_dictionary()

    compare_dictionary_columns(test_dictionary, check_dictionary)


if __name__ == "__main__":
    pytest.main([__file__])
