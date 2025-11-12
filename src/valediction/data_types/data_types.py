from __future__ import annotations

from enum import Enum


class DataType(Enum):
    TEXT = "Text"
    INTEGER = "Integer"
    FLOAT = "Float"
    DATE = "Date"
    DATETIME = "Datetime"
    FILE = "File"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value

    @classmethod
    def parse(cls, data_type: str) -> DataType:
        """Case-insensitive, forgiving parser."""
        normalised = (data_type or "").strip().lower()
        aliases = {
            "text": cls.TEXT,
            "string": cls.TEXT,
            "str": cls.TEXT,
            "int": cls.INTEGER,
            "integer": cls.INTEGER,
            "float": cls.FLOAT,
            "double": cls.FLOAT,
            "number": cls.FLOAT,
            "numeric": cls.FLOAT,
            "date": cls.DATE,
            "datetime": cls.DATETIME,
            "datetime64": cls.DATETIME,
            "timestamp": cls.DATETIME,
            "file": cls.FILE,
            "blob": cls.FILE,
            "binary": cls.FILE,
        }
        try:
            return aliases[normalised]
        except KeyError as error:
            raise ValueError(f"Unknown data type: {data_type!r}") from error

    def allows_length(self) -> bool:
        """Only TEXT should have a length attribute."""
        return self in {DataType.TEXT}

    def valid_for_primary_key(self) -> bool:
        """PKs can only be Text, Integer, Date, Datetime."""
        return self in {
            DataType.TEXT,
            DataType.INTEGER,
            DataType.DATE,
            DataType.DATETIME,
        }
