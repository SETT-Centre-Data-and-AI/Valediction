class DataDictionaryError(Exception):
    def __init__(self, message: str = "A DataDictionaryError has occurred"):
        super().__init__(message)
        self.message = message


class DuplicateHeaderError(DataDictionaryError):
    def __init__(self, message: str = "Duplicate column headers were found"):
        super().__init__(message)


class DataDictionaryImportError(Exception):
    def __init__(self, message: str = "A DataDictionaryImportError has occurred"):
        super().__init__(message)
        self.message = message


class DataDictionaryExportError(Exception):
    def __init__(self, message: str = "A DataDictionaryExportError has occurred"):
        super().__init__(message)
        self.message = message


class DataIntegrityError(Exception):
    def __init__(self, message: str = "A DataIntegrityError has occurred"):
        super().__init__(message)
        self.message = message
