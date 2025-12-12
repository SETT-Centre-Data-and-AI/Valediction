from pathlib import Path

from pandas import DataFrame

from valediction.datasets.datasets import Dataset
from valediction.dictionary.model import Dictionary


def validate(
    dataset: str | Path | dict[str, DataFrame],
    dictionary: Dictionary | str | Path,
    *,
    import_data: bool = False,
    chunk_size: int | None = 10_000_000,
    feedback: bool = True,
) -> Dataset:
    """Validate the dataset against the dictionary. Run dataset.check() afterwards to
    raise Exception if issues.

    Arguments:
        dataset (str | Path | dict[str, DataFrame]): path to CSV, DataFrame, or dictionary of table names
            to DataFrames
        dictionary (Dictionary | str | Path): dictionary to validate against as a Dictionary object
            or .xlsx filepath
        import_data (bool, optional): whether to load all data into memory. Defaults to False.
        chunk_size (int | None, optional): size of chunks for validating data to optimise RAM usage.
            Defaults to 10_000_000.
        feedback (bool, optional): whether to provide user feedback on progress. Defaults to True.

    Returns:
        Dataset: dataset, with or without Issues
    """

    _dataset: Dataset = Dataset.create_from(dataset)
    _dataset.import_dictionary(dictionary)

    if import_data:
        _dataset.import_data()

    _dataset.validate(
        chunk_size=chunk_size,
        feedback=feedback,
    )

    return _dataset
