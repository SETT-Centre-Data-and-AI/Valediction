from pathlib import Path

from valediction.demo.demo_dictionary import demo_dictionary  # noqa

DEMO_DATA = Path(__file__).resolve().parent
DEMO_DICTIONARY = DEMO_DATA / "DEMO - Data Dictionary.xlsx"
