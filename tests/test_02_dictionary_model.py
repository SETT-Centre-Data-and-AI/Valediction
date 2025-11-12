import pytest

from valediction.dictionary.model import Column, Dictionary, Table
from valediction.exceptions import DataDictionaryError
from valediction.integrity import get_config

### Helpers ###


def valid_columns() -> list[Column]:
    return [
        Column(
            name="PATIENT_HASH", order=1, data_type="text", length=12, primary_key=1
        ),
        Column(name="SAMPLE_DATE", order=2, data_type="date", primary_key=2),
        Column(name="RESULT_DATE", order=3, data_type="date"),
        Column(name="SAMPLE_TYPE", order=4, data_type="text", length=32),
        Column(name="TEST_TYPE", order=5, data_type="text", length=32, primary_key=3),
        Column(name="RESULT_RAW", order=6, data_type="text", length=256),
        Column(name="UNITS", order=7, data_type="text", length=16),
        Column(name="RESULT_NUMERIC", order=8, data_type="float"),
        Column(name="RESULT_PROCESSED", order=9, data_type="text", length=8),
        Column(name="OPERATOR", order=10, data_type="text", length=1),
        Column(name="RANGE_LOW", order=11, data_type="float"),
        Column(name="RANGE_HIGH", order=12, data_type="float"),
    ]


@pytest.fixture
def valid_table() -> Table:
    return Table(name="lab_tests", columns=valid_columns())


@pytest.fixture
def valid_dictionary(valid_table) -> Dictionary:
    return Dictionary(name="Test Data", tables=[valid_table])


def test_valid_dictionary_passes(valid_dictionary):
    assert valid_dictionary.table_count == 1
    assert valid_dictionary.column_count == 12


# Column Name Rules #
@pytest.mark.parametrize(
    "bad_name",
    [
        "1PATIENT",  # must start with a letter
        "PATIENT_",  # cannot end with underscore
        "PAT__IENT",  # double underscores not allowed
        "PAT IENT",  # whitespace not allowed
        "PAT-IENT",  # invalid char
        "TOO_LONG"
        * (get_config().max_column_name_length // len("TOO_LONG") + 1),  # length > MAX
    ],
)
def test_invalid_column_name_raises(bad_name):
    with pytest.raises(DataDictionaryError):
        Column(name=bad_name, order=1, data_type="text", length=10)


# Data Type Rules #
def test_text_without_length_raises():
    with pytest.raises(DataDictionaryError):
        Column(name="TXT_NO_LEN", order=1, data_type="text")  # length required for text


def test_length_not_allowed_for_non_text_raises():
    with pytest.raises(DataDictionaryError):
        Column(name="FLOAT_HAS_LEN", order=1, data_type="float", length=10)


def test_length_must_be_positive():
    with pytest.raises(DataDictionaryError):
        Column(name="TXT_LEN_ZERO", order=1, data_type="text", length=0)


def test_invalid_pk_type_raises():
    with pytest.raises(DataDictionaryError):
        Column(name="FLOAT_PK", order=1, data_type="float", primary_key=1)


@pytest.mark.parametrize("dtype", ["text", "integer", "date", "datetime"])
def test_valid_pk_types_pass(dtype):
    # Minimal valid definition for PK-eligible types
    kwargs = {"length": 10} if dtype == "text" else {}
    c = Column(
        name=f"PK_{dtype}".upper(), order=1, data_type=dtype, primary_key=1, **kwargs
    )
    assert c.primary_key == 1


# Order & Primary Key Constraints #
def test_order_must_be_positive_integer():
    with pytest.raises(DataDictionaryError):
        Column(name="NEG_ORDER", order=0, data_type="text", length=5)


def test_pk_must_be_positive_integer():
    with pytest.raises(DataDictionaryError):
        Column(name="PK_ZERO", order=1, data_type="text", length=5, primary_key=0)


# Table Rules #
def test_duplicate_column_name_raises():
    cols = valid_columns()
    # Duplicate name with different order
    cols.append(Column(name="PATIENT_HASH", order=99, data_type="text", length=12))
    with pytest.raises(DataDictionaryError):
        Table(name="lab_tests", columns=cols)


def test_duplicate_order_raises():
    cols = valid_columns()
    # Duplicate order with different name
    cols.append(Column(name="ANOTHER_COL", order=6, data_type="text", length=10))
    with pytest.raises(DataDictionaryError):
        Table(name="lab_tests", columns=cols)


def test_conflicting_primary_key_ordinals_raises():
    cols = valid_columns()
    # Make another column conflict with SAMPLE_DATE PK=2
    cols.append(
        Column(name="CONFLICT_PK", order=99, data_type="text", length=5, primary_key=2)
    )
    with pytest.raises(DataDictionaryError):
        Table(name="lab_tests", columns=cols)


def test_missing_primary_key_in_table_raises():
    cols = valid_columns()
    cols_no_pk = []
    for c in cols:
        # clone each Column but strip primary_key
        kwargs = dict(
            name=c.name,
            order=c.order,
            data_type=(
                c.data_type.value if hasattr(c.data_type, "value") else str(c.data_type)
            ),
            length=c.length,
            vocabulary=c.vocabulary,
            primary_key=None,
            foreign_key=c.foreign_key,
            description=c.description,
            enumerations=c.enumerations,
        )
        cols_no_pk.append(Column(**kwargs))
    with pytest.raises(DataDictionaryError):
        Table(name="lab_tests", columns=cols_no_pk)


def test_get_column_by_name_and_order(valid_table: Table):
    assert valid_table.get_column("PATIENT_HASH").order == 1
    assert valid_table.get_column(5).name == "TEST_TYPE"


def test_remove_column(valid_table: Table):
    valid_table.remove_column("OPERATOR")
    names = valid_table.get_column_names()
    assert "OPERATOR" not in names
    assert len(names) == 11


# Dictionary Rules #
def test_dictionary_counts(valid_table):
    d = Dictionary(name="Test", tables=[valid_table])
    assert d.table_count == 1
    assert d.column_count == 12


def test_dictionary_duplicate_table_raises(valid_table):
    d = Dictionary(name="Test", tables=[valid_table])
    with pytest.raises(DataDictionaryError):
        d.add_table(valid_table)


if __name__ == "__main__":
    pytest.main([__file__])
