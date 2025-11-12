REQUIRED_SHEETS = {"details", "tables", "columns", "enumerations"}

S_DETAILS = "details"
S_TABLES = "tables"
S_COLUMNS = "columns"
S_ENUMERATIONS = "enumerations"

T_TABLES = "tables"
T_COLUMNS = "columns"
T_ENUMERATIONS = "enumerations"

C_CHECKS = "checks"

DD_TABLE_MAP = {
    S_DETAILS: None,
    S_TABLES: T_TABLES,
    S_COLUMNS: T_COLUMNS,
    S_ENUMERATIONS: T_ENUMERATIONS,
}

DD_COLUMN_MAP = {
    S_DETAILS: None,
    S_TABLES: ["table", "description", C_CHECKS],
    S_COLUMNS: [
        "table",
        "column",
        "order",
        "data type",
        "length",
        "vocabularies",
        "enumerations",
        "primary key",
        "column description",
        C_CHECKS,
    ],
    S_ENUMERATIONS: ["table", "column", "code", "description", C_CHECKS],
}
