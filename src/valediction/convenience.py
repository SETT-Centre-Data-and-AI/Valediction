from pathlib import Path

from pandas import DataFrame

from valediction.datasets.datasets import Dataset
from valediction.dictionary.importing import import_dictionary
from valediction.dictionary.model import Dictionary


def validate(
    data: str | Path | dict[str, DataFrame] | Dictionary,
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
    dictionary = (
        dictionary
        if isinstance(dictionary, Dictionary)
        else import_dictionary(dictionary)
    )
    data: Dataset = Dataset.create_from(data)
    data.import_dictionary(dictionary)

    if import_data:
        data.import_data()

    data.validate(
        chunk_size=chunk_size,
        feedback=feedback,
    )

    return data
