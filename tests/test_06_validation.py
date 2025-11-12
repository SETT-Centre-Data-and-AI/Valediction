import numpy as np
import pytest

from valediction.convenience import validate
from valediction.data_types.data_types import DataType
from valediction.datasets.datasets import Dataset
from valediction.demo import DEMO_DATA, DEMO_DICTIONARY
from valediction.exceptions import DataDictionaryImportError, DataIntegrityError
from valediction.integrity import get_config, reset_default_config
from valediction.validation.issues import IssueType, Range


# Parameters
@pytest.fixture(autouse=True)
def _isolate_global_default():  # noqa
    """Ensure each test starts from a clean global default and leaves it clean."""
    reset_default_config()
    yield
    reset_default_config()


@pytest.fixture(params=[None, 10, 1_000_000], ids=["no_chunk", "chunk_10", "chunk_1m"])
def chunk_size(request: pytest.FixtureRequest) -> int:
    return request.param


# Helpers
def create_dataset() -> Dataset:
    dataset = Dataset.create_from(DEMO_DATA)
    dataset.import_dictionary(DEMO_DICTIONARY)
    return dataset


def create_dataset_imported() -> Dataset:
    dataset = create_dataset()
    dataset.import_data()
    return dataset


# Tests
def test_validation():
    # Check simple validation on Path
    dataset = create_dataset()
    dataset.validate()
    dataset.check()


def test_validation_feedback():
    # Check simple validation on Path
    dataset = create_dataset()
    dataset.validate(feedback=True)
    dataset.check()


def test_validation_import(chunk_size):
    # Check simple validation on DataFrame
    dataset = create_dataset_imported()
    dataset.import_data()
    dataset.validate(chunk_size=chunk_size)
    dataset.check()


def test_validation_raises_extra_column(chunk_size):
    dataset = create_dataset_imported()

    # Add a column, and check for errors
    dataset[0].data["TEST_EXTRA"] = "extra"

    # Checks
    with pytest.raises(DataIntegrityError):
        dataset.validate(chunk_size=chunk_size)
        dataset.check()

    issues = [
        issue
        for issue in dataset[0].validator.issues
        if issue.type == IssueType.EXTRA_COLUMN
    ]
    assert issues


def test_validation_raises_missing_column(chunk_size):
    dataset = create_dataset_imported()

    # Drop the final column, and check for errors
    df = dataset[0].data
    dataset[0].data = dataset[0].data.drop(df.columns[-1], axis=1)

    # Checks
    with pytest.raises(DataIntegrityError):
        dataset.validate(chunk_size=chunk_size)
        dataset.check()

    issues = [
        issue
        for issue in dataset[0].validator.issues
        if issue.type == IssueType.MISSING_COLUMN
    ]
    assert issues


def test_validation_raises_fully_null_column(chunk_size):
    dataset = create_dataset_imported()

    # Change the final column to nulls
    dataset[0].data[dataset[0].data.columns[-1]] = None

    # Checks
    with pytest.raises(DataIntegrityError):
        dataset.validate(chunk_size=chunk_size)
        dataset.check()

    issues = [
        issue
        for issue in dataset[0].validator.issues
        if issue.type == IssueType.FULLY_NULL_COLUMN
    ]
    assert issues


def test_validation_raises_fully_null_column_with_default_null_mixture(chunk_size):
    null_values = get_config().null_values
    dataset = create_dataset_imported()

    # Change the final column to a mixture of nulls
    dataset[0].data[dataset[0].data.columns[-1]] = np.random.choice(
        null_values, size=len(dataset[0].data)
    )

    # Checks
    with pytest.raises(DataIntegrityError):
        dataset.validate(chunk_size=chunk_size)
        dataset.check()

    issues = [
        issue
        for issue in dataset[0].validator.issues
        if issue.type == IssueType.FULLY_NULL_COLUMN
    ]
    assert issues


def test_validation_raises_pk_nulls(chunk_size):
    dataset = create_dataset_imported()
    df = dataset[0].data
    table_dictionary = dataset[0].table_dictionary
    pk_columns = table_dictionary.get_primary_keys()

    # Set the first row PKs to Null
    df.loc[0, pk_columns] = None

    # Checks
    with pytest.raises(DataIntegrityError):
        dataset.validate(chunk_size=chunk_size)
        dataset.check()

    issues = [
        issue
        for issue in dataset[0].validator.issues
        if issue.type == IssueType.PK_NULL
    ]
    assert issues
    assert issues[0].ranges == [Range(0, 0)]


def test_validation_raises_pk_collision(chunk_size):
    dataset = create_dataset_imported()
    df = dataset[0].data
    table_dictionary = dataset[0].table_dictionary
    pk_columns = table_dictionary.get_primary_keys()

    # Copy the 1st row PKs to the 2nd, and the 5th to the 6th
    df.loc[1, pk_columns] = df.loc[0, pk_columns]
    df.loc[5, pk_columns] = df.loc[4, pk_columns]

    # Checks
    with pytest.raises(DataIntegrityError):
        dataset.validate(chunk_size=chunk_size)
        dataset.check()

    issues = [
        issue
        for issue in dataset[0].validator.issues
        if issue.type == IssueType.PK_COLLISION
    ]
    assert issues
    assert issues[0].ranges == [Range(0, 1), Range(4, 5)]


def test_validation_raises_pk_whitespace(chunk_size):
    dataset = create_dataset_imported()
    df = dataset[0].data
    table_dictionary = dataset[0].table_dictionary
    pk_columns = table_dictionary.get_primary_keys()

    # Get text PK Columns

    pk_cols_text = []
    for column in table_dictionary:
        if column.name in pk_columns and column.data_type in [DataType.TEXT]:
            pk_cols_text.append(column.name)

    if not pk_cols_text:
        raise Exception(
            f"No text PK columns in table '{table_dictionary.name}' to perform test"
        )

    # Inject whitespace into the PKs
    for row_index in (0, 1, 5):
        df.loc[row_index, pk_cols_text] = (
            df.loc[row_index, pk_cols_text].astype(str).str.slice(0, 3)
            + " "
            + df.loc[row_index, pk_cols_text].astype(str).str.slice(3)
        )

    # Checks
    with pytest.raises(DataIntegrityError):
        dataset.validate(chunk_size=chunk_size)
        dataset.check()

    issues = [
        issue
        for issue in dataset[0].validator.issues
        if issue.type == IssueType.PK_WHITESPACE
    ]
    assert issues
    assert issues[0].ranges == [Range(0, 1), Range(5, 5)]


@pytest.mark.parametrize(
    "data_type", [DataType.DATE, DataType.DATETIME, DataType.INTEGER, DataType.FLOAT]
)
def test_raises_mismatch_text_to_alternate(chunk_size, data_type: DataType):
    dataset = create_dataset_imported()

    # Set Tests
    TABLE = "LAB_TESTS"
    COLUMN = "SAMPLE_TYPE"
    DATA_TYPE = data_type

    # Change DataType
    item = dataset[TABLE]
    table_dictionary = item.table_dictionary
    column = table_dictionary.get_column(COLUMN)
    column.data_type = DATA_TYPE
    column.length = None

    # Checks
    with pytest.raises(DataIntegrityError):
        dataset.validate(chunk_size=chunk_size)
        dataset.check()

    issues = [
        issue
        for issue in dataset[TABLE].validator.issues
        if issue.type == IssueType.TYPE_MISMATCH
    ]
    assert issues


def test_raises_mismatch_float_to_integer(chunk_size):
    dataset = create_dataset_imported()

    # Set Tests
    TABLE = "LAB_TESTS"
    COLUMN = "RESULT_NUMERIC"
    DATA_TYPE = DataType.INTEGER

    # Change DataType
    item = dataset[TABLE]
    table_dictionary = item.table_dictionary
    column = table_dictionary.get_column(COLUMN)
    column.data_type = DATA_TYPE
    column.length = None

    # Checks
    with pytest.raises(DataIntegrityError):
        dataset.validate(chunk_size=chunk_size)
        dataset.check()

    issues = [
        issue
        for issue in dataset[TABLE].validator.issues
        if issue.type == IssueType.TYPE_MISMATCH
    ]
    assert issues


def test_raises_mismatch_datetime_to_date(chunk_size):
    dataset = create_dataset_imported()

    # Set Tests
    TABLE = "VITALS"
    COLUMN = "OBSERVATION_TIME"
    DATA_TYPE = DataType.DATE

    # Change DataType
    item = dataset[TABLE]
    table_dictionary = item.table_dictionary
    column = table_dictionary.get_column(COLUMN)
    column.data_type = DATA_TYPE
    column.length = None

    # Checks
    with pytest.raises(DataIntegrityError):
        dataset.validate(chunk_size=chunk_size)
        dataset.check()

    issues = [
        issue
        for issue in dataset[TABLE].validator.issues
        if issue.type == IssueType.TYPE_MISMATCH
    ]
    assert issues


def test_raises_mismatch_text_to_date(chunk_size):
    dataset = create_dataset_imported()

    # Set Tests
    TABLE = "DEMOGRAPHICS"
    COLUMN = "DATE_OF_BIRTH"

    # Insert String
    dataset[TABLE].data[COLUMN] = "not_a_date"

    with pytest.raises(DataIntegrityError):
        dataset.validate(chunk_size=chunk_size)
        dataset.check()

    issues = [
        issue
        for issue in dataset[TABLE].validator.issues
        if issue.type == IssueType.TYPE_MISMATCH
    ]
    assert issues


def test_raises_text_too_long(chunk_size):
    dataset = create_dataset_imported()

    # Set Tests
    TABLE = "LAB_TESTS"
    COLUMN = "PATIENT_HASH"

    # Shorten Length
    item = dataset[TABLE]
    table_dictionary = item.table_dictionary
    column = table_dictionary.get_column(COLUMN)
    column.length = column.length - 3

    # Checks
    with pytest.raises(DataIntegrityError):
        dataset.validate(chunk_size=chunk_size)
        dataset.check()

    issues = [
        issue
        for issue in dataset[TABLE].validator.issues
        if issue.type == IssueType.TEXT_TOO_LONG
    ]
    assert issues


def test_raises_forbidden_characters(chunk_size):
    dataset = create_dataset_imported()
    config = get_config()
    config.forbidden_characters = ["^", "(", ")"]

    # Set Tests
    TABLE = "LAB_TESTS"
    COLUMN = "SAMPLE_TYPE"

    # Add Forbidden Characters
    df = dataset[TABLE].data
    df[COLUMN] = np.random.choice(config.forbidden_characters, size=len(df[COLUMN]))

    # Checks
    with pytest.raises(DataIntegrityError):
        dataset.validate(chunk_size=chunk_size)
        dataset.check()

    issues = [
        issue
        for issue in dataset[TABLE].validator.issues
        if issue.type == IssueType.FORBIDDEN_CHARACTER
    ]
    assert issues


def test_passes_without_forbidden_characters(chunk_size):
    dataset = create_dataset_imported()
    config = get_config()
    config.forbidden_characters = ["^"]
    replacement = ["x"]

    # Set Tests
    TABLE = "LAB_TESTS"
    COLUMN = "SAMPLE_TYPE"

    # Add Forbidden Characters
    df = dataset[TABLE].data
    df[COLUMN] = np.random.choice(replacement, size=len(df[COLUMN]))

    # Checks
    dataset.validate(chunk_size=chunk_size)
    dataset.check()

    issues = [
        issue
        for issue in dataset[TABLE].validator.issues
        if issue.type == IssueType.FORBIDDEN_CHARACTER
    ]
    assert not issues


def test_validate_import(chunk_size):
    dataset = create_dataset()
    dataset.validate(chunk_size=chunk_size)
    dataset.import_data()  # will auto apply dictionary
    dataset.check()


def test_import_validate_apply(chunk_size):
    dataset = create_dataset()
    dataset.validate(chunk_size=chunk_size)
    dataset.apply_dictionary()


def test_raises_validate_without_dictionary():
    dataset = create_dataset()
    object.__setattr__(dataset[0], "table_dictionary", None)

    with pytest.raises(DataDictionaryImportError):
        dataset.validate()
        dataset.check()


def test_raises_apply_without_validation():
    dataset = create_dataset()

    with pytest.raises(DataIntegrityError):
        dataset.apply_dictionary()


def test_validate_convenience(chunk_size):
    dataset = validate(
        data=DEMO_DATA, dictionary=DEMO_DICTIONARY, chunk_size=chunk_size
    )
    dataset.check()


def test_validate_convenience_with_import(chunk_size):
    dataset = validate(
        data=DEMO_DATA,
        dictionary=DEMO_DICTIONARY,
        chunk_size=chunk_size,
        import_data=True,
    )
    dataset.check()


def test_validate_convenience_raises_error(chunk_size):
    config = get_config()
    config.forbidden_characters = ["a"]
    dataset = validate(
        data=DEMO_DATA,
        dictionary=DEMO_DICTIONARY,
        chunk_size=chunk_size,
        import_data=True,
    )
    with pytest.raises(DataIntegrityError):
        dataset.check()

    issues = [
        issue for issue in dataset.issues if issue.type == IssueType.FORBIDDEN_CHARACTER
    ]
    assert issues


if __name__ == "__main__":
    pytest.main([__file__])
