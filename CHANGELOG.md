## v1.8.3 (2026-10-14)

### Feat

- **optimisation**: performance optimisation for new fixes

### Fix

- **warnings**: hardened  invalid_mask_integer_out_of_range() against overflows in scientific notation
- **warning**: null validation differences between CSV and DataFrame differences
- **warning**: nanoseconds incorporated into datetime vs date checks
- **warning**: fixed fmt_or_none is None in datetime format matching
- **infer_datetime-&-decimal-validation**: fixed warnings around early datetime format inference and integer acceptance hardening

## v1.8.2 (2026-10-14)

### Feat

- **convenience**: refactored convenience validate() function to add Dataset pass, supporting Cynric downstream without code duplication

## v1.8.1 (2026-10-14)

### Feat

- **fail_fast**: tweaked progress bar to not show completed
- **fail_fast**: added fail_fast argument to validation functions to exit and fail immediately on first error

## v1.8.0 (2026-10-13)

### Feat

- **INCONSISTENT_DATE**: added issue handling for inconsistent day-first/month-first issue handling
- **timestamps**: implemented timestamp/date fixes
- **validation**: removed global conversion to string as non-performative
- **validation**: implemented conversion to string prior to validation, ready for performance testing
- **validation**: added opt-in `fail_fast` validation with partial issue reporting

### Fix

- **validation\helpers.py**: fixed passing non-midnight datetimes as dates when dtypes are mixed in column
- **REVERSION**: reverted to v1.7.5 codebase to correct performance hits following timestamp changes

## v1.7.8 (2026-10-05) [YANKED]
_yanked due to significant reduction in speed when processing datetimes, fixed in v1.8.0_

### Feat

- **data_type_helpers.py**: added timezone partitioning to future-proof against Pandas' mixed timezone FutureWarning
- **_parse_integer**: added cache limit before clearing to prevent endless cache growth while validating
- **INCONSISTENT_DATE**: report valid unambiguous day-first and month-first date values mixed in one DATE or TIMESTAMP column
- **dates**: dates accepted as timestamp types
- **io**: preserve typed values across CSV import and export
- **timestamp**: fix for locally imported data timestamp interpretation
- **dates**: dates accepted as timestamp types
- **generation.py**: added check for duplicate headers to raise early error with message, rather than erroring on nested DataFrames

## v1.7.6 (2026-08-14) [YANKED]
_yanked due to significant reduction in speed when processing datetimes, fixed in v1.8.0_
### Feat

- **generation.py**: added check for duplicate headers to raise early error with message, rather than erroring on nested DataFrames

## v1.7.5 (2026-06-16)

### Feat

- **bigint**: allow_bigint behaviour adhered to when generating dictionary as well as validating

## v1.7.4 (2026-06-16)

### Fix

- **config**: set enforce_no_null_columns to False by default
- **issues**: update return types for better clarity and type checking
- **validation**: conditionally check for fully null columns based on config
- **type_inference, datasets, progress**: add type hints for better clarity and type checking
- **dev_dependencies**: add types-tqdm to development dependencies
- **dependencies**: update package versions

## v1.7.3 (2026-05-26)

### Fix

- **dictionary_generation**: fix for timezone-aware datetimes

## v1.7.2 (2026-03-24)

## v1.7.1 (2026-03-23)

### Feat

- **validation**: staging optimisation in new branch + tweaked docs

### Fix

- **config.yaml**: attempt to fix smoke tests
- **workflow-orchestrator.yaml**: fixed issue in write permissions

## v1.7.0 (2026-03-09)

### Feat

- **validation**: significant optimisation when not chunking by removing the need to hash PKs

## v1.6.0 (2026-03-05)

### Feat

- **convenience**: added convenient create_dataset function
- **validation**: added pk_col_max_length support with default = off
- **demo**: added create_demo_dataset function

## v1.5.0 (2026-02-03)

### Feat

- **bigint_checking**: wired in `allow_bigint` optional checking

### Refactor

- **validation**: refactored to simplify _check_column_types

## v1.4.0 (2026-02-03)

### Feat

- **datetime/timestamp**: recoded dictionary & valediction to use 'Timestamp'

### Fix

- **type_inference**: fixed attempts to trim non-text columns in dicti… (#20)

## v1.3.0 (2026-02-02)

### Fix

- **dictionary/importing.py**: fixed column_description import bring skipped
- **type_inference**: fixed attempts to trim non-text columns in dictionary generation type inference

## v1.2.0 (2025-12-14)

### Feat

- **integrity**: added type checking to inject_config_variables() function
- **integrity**: implemented config variable injection for external packages like Cynric and associated tests
- **case-insensitivity**: major refactoring to allow case-insensitive dictionary & data import, lookup, and validation
- **validation**: hardened str accessors to allow dict[str. DataFrame] to be validated without forcing to dtype str
- **model**: hardened dictionary with type checks

## v1.1.0 (2025-12-12)

### Fix

- **tests**: aligned tests with validation fix

## v1.0.4 (2025-12-12)

### Fix

- **convenience**: fixed dataset argument in validate

## v1.0.3 (2025-12-01)

## v1.0.2 (2025-12-01)

### Fix

- **integrity.py**: fixed case-sensitive template DD path for Linux

## v1.0.0 (2025-12-01)
