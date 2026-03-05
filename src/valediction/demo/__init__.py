from pathlib import Path

from valediction.datasets.datasets import Dataset  # noqa
from valediction.demo.demo_dictionary import demo_dictionary  # noqa

DEMO_DATA = Path(__file__).resolve().parent
DEMO_DICTIONARY = DEMO_DATA / "DEMO - Data Dictionary.xlsx"


def create_demo_dataset() -> Dataset:
    dataset = Dataset.create_from(DEMO_DATA)
    dataset.import_dictionary(DEMO_DICTIONARY)
    return dataset
