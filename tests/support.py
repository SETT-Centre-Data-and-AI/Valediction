from valediction.dictionary.model import Dictionary
from valediction.support import _normalise


def compare_dictionary_tables(
    test_dictionary: Dictionary, check_dictionary: Dictionary
):
    for test_table in test_dictionary:
        table_name = test_table.name
        check_table = check_dictionary.get_table(table_name)

        error_leader = f"Table '{test_table.name}' testing vs checker mismatch:"

        assert _normalise(test_table.name) == _normalise(check_table.name), (
            f"{error_leader}\nname: {test_table.name} != {check_table.name}"
        )

        assert test_table.description == check_table.description, (
            f"{error_leader}\n"
            f"description: {test_table.description} != {check_table.description}"
        )


def compare_dictionary_columns(
    test_dictionary: Dictionary, check_dictionary: Dictionary
):
    for test_table in test_dictionary:
        table_name = test_table.name
        check_table = check_dictionary.get_table(table_name)

        for test_column in test_table:
            column_name = test_column.name
            check_column = check_table.get_column(column_name)
            error_leader: str = f"Table '{test_table.name}' Column '{test_column.name}' testing vs checker mismatch:"

            for annotation in [
                "name",
                "data_type",
                "length",
                "primary_key",
                "foreign_key",
                "description",
                "enumerations",
            ]:
                assert getattr(test_column, annotation) == getattr(
                    check_column, annotation
                ), (
                    f"{error_leader}\n{annotation}: {getattr(test_column, annotation)} != {getattr(check_column, annotation)}"
                )
