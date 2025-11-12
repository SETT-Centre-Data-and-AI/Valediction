from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import pytest

from valediction import Dataset, demo
from valediction.datasets.datasets import DatasetItem

EXPORT_DIR = Path(__file__).parent


# Get Helpers
@contextmanager
def cleanup(items: DatasetItem | Iterable[DatasetItem]):
    items: list[DatasetItem] = [items] if isinstance(items, DatasetItem) else items

    paths = [EXPORT_DIR / f"{item.name}.csv" for item in items]

    # Pre-clean
    for p in paths:
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    try:
        yield paths[0] if len(paths) == 1 else paths
    finally:
        for p in paths:
            try:
                p.unlink()
            except FileNotFoundError:
                pass


def export_exists(item: DatasetItem):
    path = EXPORT_DIR / f"{item.name}.csv"
    return path.exists()


def get_dataset() -> Dataset:
    dataset = Dataset.create_from(demo.DEMO_DATA)
    dataset.import_dictionary(demo.DEMO_DICTIONARY)
    return dataset


# Tests at item level
def test_item_export_post_validation() -> None:
    dataset = get_dataset()
    dataset.import_data()
    dataset.validate()
    item = dataset[0]

    with cleanup(item):
        item.export_data(directory=EXPORT_DIR)


def test_raises_item_export_no_validation() -> None:
    dataset = get_dataset()
    dataset.import_data()
    item = dataset[0]

    with cleanup(item):
        with pytest.raises(ValueError):
            item.export_data(directory=EXPORT_DIR)


def test_raises_item_export_no_import() -> None:
    dataset = get_dataset()
    item = dataset[0]

    with cleanup(item):
        with pytest.raises(ValueError):
            item.export_data(directory=EXPORT_DIR)


def test_raises_item_export_validated_but_no_import() -> None:
    dataset = get_dataset()
    dataset.validate()
    item = dataset[0]

    with cleanup(item):
        with pytest.raises(ValueError):
            item.export_data(directory=EXPORT_DIR)


def test_raises_item_export_imported_but_not_validated() -> None:
    dataset = get_dataset()
    dataset.import_data()
    item = dataset[0]

    with cleanup(item):
        with pytest.raises(ValueError):
            item.export_data(directory=EXPORT_DIR)


def test_item_export_validation_override() -> None:
    dataset = get_dataset()
    dataset.import_data()
    item = dataset[0]

    with cleanup(item):
        item.export_data(directory=EXPORT_DIR, enforce_validation=False)


def test_item_export_raises_on_conflict() -> None:
    dataset = get_dataset()
    dataset.import_data()
    dataset.validate()
    item = dataset[0]

    with cleanup(item):
        item.export_data(directory=EXPORT_DIR)
        with pytest.raises(ValueError):
            item.export_data(directory=EXPORT_DIR, enforce_validation=False)


def test_item_export_overwrite() -> None:
    dataset = get_dataset()
    dataset.import_data()
    dataset.validate()
    item = dataset[0]

    with cleanup(item):
        item.export_data(directory=EXPORT_DIR)
        item.export_data(directory=EXPORT_DIR, enforce_validation=False, overwrite=True)


# Tests at dataset level
def test_dataset_export_post_validation() -> None:
    dataset = get_dataset()
    dataset.import_data()
    dataset.validate()

    with cleanup(dataset):
        dataset.export_data(directory=EXPORT_DIR)


def test_dataset_skips_unvalidated() -> None:
    dataset = get_dataset()
    dataset.import_data()
    indexes = [0, 1]

    validated = [dataset[i] for i in indexes]
    unvalidated = [dataset[i] for i in range(len(dataset)) if i not in indexes]

    for item in validated:
        item.validate()

    with cleanup(dataset):
        dataset.export_data(directory=EXPORT_DIR)
        for item in validated:
            assert export_exists(item)
        for item in unvalidated:
            assert not export_exists(item)


def test_dataset_export_skips_unimported() -> None:
    dataset = get_dataset()
    dataset.validate()
    indexes = [0, 1]

    imported = [dataset[i] for i in indexes]
    unimported = [dataset[i] for i in range(len(dataset)) if i not in indexes]

    for item in imported:
        item.import_data()

    with cleanup(dataset):
        dataset.export_data(directory=EXPORT_DIR)

        for item in imported:
            assert export_exists(item)
        for item in unimported:
            assert not export_exists(item)


def test_dataset_export_validation_override() -> None:
    dataset = get_dataset()
    dataset.import_data()

    with cleanup(dataset):
        dataset.export_data(directory=EXPORT_DIR, enforce_validation=False)

        for item in dataset:
            assert export_exists(item)


def test_dataset_export_raises_on_conflict() -> None:
    dataset = get_dataset()
    dataset.import_data()
    dataset.validate()

    with cleanup(dataset):
        dataset.export_data(directory=EXPORT_DIR)

        for item in dataset:
            assert export_exists(item)

        with pytest.raises(ValueError):
            dataset.export_data(directory=EXPORT_DIR)


def test_dataset_export_overwrite() -> None:
    dataset = get_dataset()
    dataset.import_data()
    dataset.validate()

    with cleanup(dataset):
        dataset.export_data(directory=EXPORT_DIR)

        for item in dataset:
            assert export_exists(item)

        dataset.export_data(directory=EXPORT_DIR, overwrite=True)


if __name__ == "__main__":
    pytest.main([__file__])
