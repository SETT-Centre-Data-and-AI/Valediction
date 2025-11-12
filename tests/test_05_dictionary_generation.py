import pytest

from valediction.datasets.datasets import Dataset
from valediction.demo import DEMO_DATA, demo_dictionary

DICT_NAME = "TEST"


# Parameters
@pytest.fixture(params=[True, False], ids=["feedback", "no_feedback"])
def feedback(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(params=[True, False], ids=["debug", "no_debug"])
def debug(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(
    params=[None, 100, 1_000, 1_000_000],
    ids=["no_chunk", "chunk_100", "chunk_1k", "chunk_1m"],
)
def chunk_size(request: pytest.FixtureRequest) -> int | None:
    return request.param


# Helpers
def create_dataset() -> Dataset:
    dataset = Dataset.create_from(DEMO_DATA)
    return dataset


def test_generate_dictionary(feedback) -> None:
    dataset = create_dataset()
    dictionary = dataset.generate_dictionary(
        dictionary_name=DICT_NAME, feedback=feedback
    )

    assert dictionary
    assert len(dictionary) == 4


def test_generated_dictionary_integrity(feedback) -> None:
    dataset = Dataset.create_from(DEMO_DATA)
    demo_dict = demo_dictionary()
    gen_dict = dataset.generate_dictionary("Synthetic Data")

    # Check tables
    assert len(gen_dict) == len(demo_dict)
    assert set([table.name for table in gen_dict]) == set(
        table.name for table in gen_dict
    )

    # Check Columns
    for gen_table in gen_dict:
        demo_table = demo_dict.get_table(table=gen_table.name)
        assert len(gen_table) == len(demo_table)
        assert set([column.name for column in gen_table]) == set(
            column.name for column in demo_table
        )

        # Check Data Types
        for column in gen_table:
            assert (
                column.data_type == demo_table.get_column(column=column.name).data_type
            )


def test_runtimes_always_saved(feedback) -> None:
    dataset = create_dataset()
    _dictionary = dataset.generate_dictionary(
        dictionary_name=DICT_NAME, feedback=feedback
    )

    assert (item._dictionary_runtimes for item in dataset)


def test_with_sample_rows(feedback) -> None:
    dataset = create_dataset()
    _dictionary = dataset.generate_dictionary(
        dictionary_name=DICT_NAME, feedback=feedback, sample_rows=100
    )

    assert (item._dictionary_runtimes for item in dataset)


def test_with_chunk_size(feedback, chunk_size) -> None:
    dataset = create_dataset()
    _dictionary = dataset.generate_dictionary(
        dictionary_name=DICT_NAME, feedback=feedback, chunk_size=chunk_size
    )

    assert (item._dictionary_runtimes for item in dataset)


def test_with_sample_rows_and_chunk_size(feedback, chunk_size) -> None:
    dataset = create_dataset()
    _dictionary = dataset.generate_dictionary(
        dictionary_name=DICT_NAME,
        feedback=feedback,
        chunk_size=chunk_size,
        sample_rows=100,
    )

    assert (item._dictionary_runtimes for item in dataset)


def test_with_debug(feedback, chunk_size, debug) -> None:
    dataset = create_dataset()
    _dictionary = dataset.generate_dictionary(
        dictionary_name=DICT_NAME,
        feedback=feedback,
        chunk_size=chunk_size,
        sample_rows=100,
        debug=debug,
    )

    assert (item._dictionary_runtimes for item in dataset)


if __name__ == "__main__":
    pytest.main([__file__])
