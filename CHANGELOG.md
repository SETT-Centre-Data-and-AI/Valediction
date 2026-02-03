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
