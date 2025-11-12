import pytest

from valediction.convenience import validate as _validate
from valediction.data_types.data_types import DataType
from valediction.datasets.datasets import Dataset
from valediction.demo import DEMO_DATA, DEMO_DICTIONARY, demo_dictionary
from valediction.dictionary.importing import import_dictionary
from valediction.dictionary.model import Dictionary
from valediction.exceptions import DataDictionaryError, DataIntegrityError
from valediction.integrity import Config, get_config, reset_default_config
from valediction.validation.issues import IssueType


# Parameters
@pytest.fixture(autouse=True)
def _isolate_global_default():  # noqa
    """Ensure each test starts from a clean global default and leaves it clean."""
    reset_default_config()
    yield
    reset_default_config()


# Helpers
def get_dictionary() -> Dictionary:
    return demo_dictionary()


def build_dataset() -> Dataset:
    dataset = Dataset.create_from(DEMO_DATA)
    dictionary = get_dictionary()
    dataset.import_dictionary(dictionary)
    return dataset


def validate() -> Dataset:
    return _validate(data=DEMO_DATA, dictionary=demo_dictionary())


# Test Dictionaries
def test_dictionary_default_config():
    _ = get_config()
    dictionary = get_dictionary()
    dictionary.check()


def test_import_dictionary_default_config():
    _ = get_config()
    dictionary = import_dictionary(DEMO_DICTIONARY)
    dictionary.check()


def test_table_name_too_long():
    config = get_config()
    config.max_table_name_length = 1

    with pytest.raises(DataDictionaryError):
        validate()


def test_column_name_too_long():
    config = get_config()
    config.max_column_name_length = 1
    with pytest.raises(DataDictionaryError):
        validate()


def test_too_many_primary_keys():
    config = get_config()
    config.max_primary_keys = 1
    with pytest.raises(DataDictionaryError):
        validate()


def test_invalid_name_pattern():
    config = get_config()
    config.invalid_name_pattern = r"[^onlythesecharsarevalid]"
    with pytest.raises(DataDictionaryError):
        validate()


# Test Validation
def test_validate_default_config():
    validate()


def test_validate_default_nulls_raises():
    TABLE = "DEMOGRAPHICS"
    COLUMN = "SEX"
    OPTIONS = ["", "Male", "Female", "Other"]

    config = get_config()
    config.null_values = OPTIONS

    dataset = build_dataset()
    dataset.import_data()
    dataset.validate()

    with pytest.raises(DataIntegrityError):
        dataset.check()

    issues = [
        issue
        for issue in dataset[TABLE].validator.issues
        if issue.type == IssueType.FULLY_NULL_COLUMN and issue.column == COLUMN
    ]
    assert issues


def test_validate_forbidden_characters_raises():
    config = get_config()
    config.forbidden_characters = [
        "a",
        "b",
        "c",
        "d",
        "e",
    ]

    dataset = build_dataset()
    dataset.import_data()
    dataset.validate()

    with pytest.raises(DataIntegrityError):
        dataset.check()

    issues = [
        issue for issue in dataset.issues if issue.type == IssueType.FORBIDDEN_CHARACTER
    ]
    assert issues


def test_validate_date_formats_raises():
    config = get_config()
    config.date_formats = {"%Y-%m-%d": DataType.DATE}

    dataset = build_dataset()
    dataset.import_data()
    dataset.validate()

    with pytest.raises(DataIntegrityError):
        dataset.check()

    issues = [
        issue for issue in dataset.issues if issue.type == IssueType.TYPE_MISMATCH
    ]
    assert issues


def test_context_manager():
    default_chars = Config().forbidden_characters
    new_characters = default_chars + ["test_chars"]

    # Check Tweak
    with Config() as config:
        config.forbidden_characters = new_characters
        assert config.forbidden_characters == new_characters

    # Check Reset
    assert get_config().forbidden_characters == default_chars


def test_validate_with_context_manager():
    default_chars = Config().forbidden_characters
    with Config() as config:
        config.forbidden_characters = [
            "a",
            "b",
            "c",
            "d",
            "e",
        ]

        dataset = build_dataset()
        dataset.import_data()
        dataset.validate()

        with pytest.raises(DataIntegrityError):
            dataset.check()

        issues = [
            issue
            for issue in dataset.issues
            if issue.type == IssueType.FORBIDDEN_CHARACTER
        ]
        assert issues

    # Check Reset
    assert get_config().forbidden_characters == default_chars


if __name__ == "__main__":
    pytest.main([__file__])
