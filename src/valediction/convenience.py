from pathlib import Path

from pandas import DataFrame

from valediction.datasets.datasets import Dataset
from valediction.dictionary.model import Dictionary
from valediction.exceptions import DataDictionaryError, DataIntegrityError


def create_dataset(
    data: str | Path | dict[str, DataFrame], dictionary: Dictionary | Path | str = ""
) -> Dataset:
    """Build a Dataset from a path (file/dir of files) or dictionary of {name:
    DataFrame}.

    Args:
        dataset (str | Path | dict[str, DataFrame]): Path to file or directory of files,
            or dictionary of existing DataFrame objects {name: DataFrame}.
        dictionary (Dictionary | Path | str, optional): Dictionary object or filepath
            to Valediction dictionary to attach. Defaults to "".

    Returns:
        Dataset: Valediction Dataset object
    """
    dataset = Dataset.create_from(dataset=data)
    if dictionary != "":
        dataset.import_dictionary(dictionary)

    return dataset


def validate(
    dataset: str | Path | dict[str, DataFrame] | Dataset,
    dictionary: Dictionary | str | Path | None = None,
    *,
    import_data: bool = False,
    chunk_size: int | None = 10_000_000,
    feedback: bool = True,
    fail_fast: bool = False,
    apply_dtypes: bool = True,
) -> Dataset:
    """Validate the dataset against the dictionary. Run `dataset.check()` afterwards to
    raise an Exception if issues. If `fail_fast` is True, validation will stop at the
    first issue-producing check and raise an Exception immediately.

    Arguments:
        dataset (str | Path | dict[str, DataFrame] | Dataset): path to CSV, DataFrame, dictionary of table names
            to DataFrames, or existing Dataset object
        dictionary (Dictionary | str | Path | None): dictionary to validate against as a Dictionary object
            or .xlsx filepath. Argument is optional only if a Valediction Dataset with a dictionary already
            attached is passed to `dataset`.
        import_data (bool, optional): whether to load all data into memory. Defaults to False.
        chunk_size (int | None, optional): size of chunks for validating data to optimise RAM usage.
            Defaults to 10_000_000.
        feedback (bool, optional): whether to provide user feedback on progress. Defaults to True.
        fail_fast (bool, optional): stop after the first issue-producing check and raise
            with the partial Dataset and issues. Defaults to False.
        apply_dtypes (bool, optional): Apply dictionary dtypes to in-memory data after
            successful validation. This option is intended for downstream extensions that need
            to defer conversion until extended validation is complete. It is not recommended
            to change this value in typical usage. Defaults to True.

    Returns:
        Dataset: dataset, with or without Issues
    """

    _dataset = dataset if isinstance(dataset, Dataset) else Dataset.create_from(dataset)

    if dictionary is not None:
        _dataset.import_dictionary(dictionary)
    else:
        if _dataset.dictionary is None:
            raise DataDictionaryError(
                "A dictionary argument must be provided when passing in a path to a dataset "
                "or a dict of DataFrames. If a pre-prepared Dataset is passed without a "
                "Dictionary already attached, this must also be provided."
            )

    if import_data:
        _dataset.import_data(apply_dtypes=apply_dtypes)

    try:
        _dataset.validate(
            chunk_size=chunk_size,
            feedback=feedback,
            fail_fast=fail_fast,
            apply_dtypes=apply_dtypes,
        )
    except DataIntegrityError as error:
        if fail_fast:
            error.dataset = _dataset
        raise

    return _dataset
