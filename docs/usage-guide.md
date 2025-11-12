# Usage Guide
[Back to Readme](../README.md)

Valediction centres on two key objects: `Dataset` and `Dictionary` objects. The following examples will use a Jupyter Notebook, and the demo dataset & accompanying dictionary, stored in `/valediction/demo/`. These comprise four tables: 'DEMOGRAPHICS', 'DIAGNOSES', 'LAB_TESTS' and 'VITALS'.

Note that the template `.xltx` compatible with Valediction is stored in `/valediction/dictionary/template`.

## 📚 Contents <a id="contents"></a>
- [Configuration](#configuration)
- [Datasets](#datasets)
- [Data Dictionary IO](#data_dictionary_io)
- [Data Dictionary Creation](#data_dictionary_creation)
- [Data Validation](#data_validation)
- [Issue Inspection](#issue_inspection)
- [Convenience](#convenience)
- [Full Run](#full_run)

## 🧩 Configuration <a id="configuration"></a>
Valediction runs a number of checks to enforce data integrity. Some of these are configurable via a globally-defined configuration, including:
 - **Max table name length**: maximum table name character length (default = 63)
 - **Max column name length**: maximum column name character length (default = 30)
 - **Max primary keys**: maximum number of primary keys (default = 7)
 - **Invalid column/table name patterns**: regex for invalid characters in names (default = numbers/characters/underscores only)
 - **Null values**: case-inensitive values to interpret as nulls during validation (default = "", "null", "none")
 - **Forbidden characters**: characters disallowed in values (default = no characters forbidden)
 - **Date Formats**: acceptable date formats, and their classification as DATE or DATETIME

These can be accessed and amended to set new global defaults - for example to allow longer table/column names:

```python
import valediction as vale
config = vale.get_config()
display(config) # view settings
```
```python
# Extend maximum lengths for PostgreSQL
config.max_table_name_length = 63
config.max_column_name_length = 63
# These settings are amended globally
```

To reset default configuration settings, run the below. 

```python
vale.reset_default_config()
```

Configuration can also be temporarily set with a context manager, before reverting to original defaults:
```python
with vale.Config() as config:
    config.max_table_name_length = 63
    config.max_column_name_length = 63
    
    # do stuff
    ...
    ...
    
# Config resets outside the block to original defaults
```

## 🧮 Datasets <a id="datasets"></a>
`Dataset`s are the central object in Valediction. They comprise either `DataFrame`s or paths pointing to CSV files (to be read in chunks when required so as to optimise RAM usage), and also contain a loaded `Dictionary` with which data can be validated. A `Dataset` contains multiple `DatasetItem` objects – each of which represents a separate table. Valediction is case-insensitive, so tables can be called without matching case.

Much of Valediction's overall functionality is accessible directly from a `Dataset`. These include:
 - `dataset.import_dictionary()`
 - `dataset.export_dictionary()`
 - `dataset.generate_dictionary()`
 - `dataset.validate()`
 - `dataset.import_data()`

### Creating from CSVs
The simplest method of establishing a `Dataset` is to use the `Dataset.create_from()` class constructor, and passing a path to a folder containing CSVs. The created `Dataset` object is iterable, and will contain `DatasetItem` objects that point to the CSV files. This allows almost instantaneous creation of the `Dataset`, with data ready to be loaded on demand, as required. An example using the demo dataset is provided below:

```python
from valediction import demo, Dataset

# Create the Dataset
path_to_data = demo.DEMO_DATA
dataset = Dataset.create_from(path_to_data)
dataset
```
```
Dataset(len=4, dictionary_loaded=True, 
 - DatasetItem(name='DEMOGRAPHICS', data=Path('DEMOGRAPHICS.csv'), validated=False)
 - DatasetItem(name='DIAGNOSES', data=Path('DIAGNOSES.csv'), validated=False)
 - DatasetItem(name='LAB_TESTS', data=Path('LAB_TESTS.csv'),  validated=False)
 - DatasetItem(name='VITALS', data=Path('VITALS.csv'), validated=False)
)
```

These tables can be explored by calling index (`dataset[0]`) or name (`dataset['demographics']`). Data (which may be either a `Path` or `DataFrame`) can be inspected as follows:

```python
demographics = dataset['demographics']
demographics.data
```
```
WindowsPath('C:/.../valediction/src/valediction/demo/DEMOGRAPHICS.csv')
```

### Importing vs Pointing
Once a `Dataset` is established, data can either be loaded via `dataset.import_data()`, or left with each `DatasetItem` pointing to the csv. In the latter case, downstream dictionary generation or data validation can take advantage of chunking to optimise RAM usage, whereby data are processed in batches and disposed of. This may be preferable when processing enormous datasets.

### Direct Construction
If integrated within an existing pipeline where `DataFrame` objects are already imported, a Dataset can be established by passing in a dict of _name: `DataFrame`_ pairs (note that while import from CSV allows Valediction to assign names based on filenames, this is not possible with a `DataFrame`). An example is given below:

```python
from valediction import demo, Dataset

# Ready-made DataFrames
path_to_data = demo.DEMO_DATA
demographics = path_to_data / "DEMOGRAPHICS.csv"
diagnoses = path_to_data / "DIAGNOSES.csv"
lab_tests = path_to_data / "LAB_TESTS.csv"
vitals = path_to_data / "VITALS.csv"

# Construct dataset directly
dataset = Dataset.create_from({
    "DEMOGRAPHICS": demographics,
    "DIAGNOSES": diagnoses,
    "LAB_TESTS": lab_tests,
    "VITALS": vitals,
})
dataset
```
```
Dataset(len=4, dictionary_loaded=False, 
 - DatasetItem(name='DEMOGRAPHICS', data=Path('DEMOGRAPHICS.csv'), validated=False)
 - DatasetItem(name='DIAGNOSES', data=Path('DIAGNOSES.csv'), validated=False)
 - DatasetItem(name='LAB_TESTS', data=Path('LAB_TESTS.csv'), validated=False)
 - DatasetItem(name='VITALS', data=Path('VITALS.csv'), validated=False)
)
```


## 🗂️ Data Dictionary IO <a id="data_dictionary_io"></a>
The `Dictionary` – whether imported, generated, or manually created – contains the key schema metadata with which to validate the `Dataset`. This package contains a compatible `.xltx` MS Excel template ([here](../src/valediction/dictionary/template/PROJECT%20-%20Data%20Dictionary.xltx)). Dictionaries are fully interoperable, allowing:

 - Import from Excel to Python
 - Generation from a `Dataset` in Python and export to Excel
 - Manual construction in Python and export to Excel

Dictionaries contain metadata around tables and columns, including:
 - Names
 - Descriptions
 - Column order
 - Data types
 - Primary key assignment
 - Lengths (for text columns)
 - Enumerations (decodable column values) - _not currently enforced in validation_
 - Foreign key targets - _not currently enforced in validation_

The `Dictionary` comprises nested iterable lists of tables and columns, indexable by index (`Dictionary[2][0]`) or name (`Dictionary['LAB_TESTS']['PATIENT_ID']`).

### Importing a Dictionary
Often, the easiest method of constructing a dictionary is to build it in Excel and import it. Using the [template](../src/valediction/dictionary/template/PROJECT%20-%20Data%20Dictionary.xltx), create an `.xlsx` file, noting incomplete/error warnings.

A `Dictionary` can then be loaded in:

```python
import valediction as vale
from valediction import demo

path_to_dictionary = demo.DEMO_DICTIONARY
dictionary = vale.import_dictionary(path_to_dictionary)
```

Tables and columns can be called by index, or by name:
```python
display(dictionary['DEMOGRAPHICS'])
patient_hash = dictionary['DEMOGRAPHICS']['PATIENT_HASH']
```
```
Table(name='DEMOGRAPHICS', description='Demographic information for synthetic patients', 
 - Column(name='PATIENT_HASH', order=1, data_type='Text(12)', primary_key=1)
 - Column(name='DATE_OF_BIRTH', order=2, data_type='Date')
 - Column(name='ETHNICITY', order=3, data_type='Text(1)')
 - Column(name='SEX', order=4, data_type='Text(6)')
 - Column(name='DATE_OF_DEATH', order=5, data_type='Date'))
```

Tweaks and additions to the dictionary objects can be made in place.

### Exporting a Dictionary
Exporting a dictionary (whether manually created, generated from data, or imported and then tweaks) is also straightforward. When exporting, filenames can be set or automatically generated, and files can be overwritten.

```python
import valediction as vale
from valediction import demo

# Set paths
path_to_dictionary = demo.DEMO_DICTIONARY
path_export = "" # here, exporting to the same folder as the notebook

# Import dictionary
dictionary = vale.import_dictionary(path_to_dictionary)

# Example tweak and check
dictionary['DIAGNOSES']['DIAGNOSIS_CODE'].length = 10
display(dictionary['DIAGNOSES'])

# Re-export
vale.export_dictionary(dictionary, directory = path_export)
```


## 📖 Data Dictionary Creation <a id="data_dictionary_creation"></a>
### Manual Creation
Data dictionaries can be manually created in Python, allowing a user to set table names and descriptions, and column data, including:
 - name
 - description
 - order (in table)
 - data_type (text, integer, float, date, or datetime)
 - length (if text)
 - primary_key (numeric order within key)
 - enumerations (as a dict of code: value pairs)
 - foreign_key (format = 'TABLE.COLUMN')

Enumerations and foreign keys are not currently enforced during validation. Primary keys and column orders cannot conflict within a single table. Length can only be applied to text columns. All table/column names are normalised to uppercase.

Here, we create a dictionary with the demo dataset's DIAGNOSIS table:

```python
from valediction import Dictionary, Table, Column, reset_default_config
reset_default_config()

diagnoses = Table(
    name="diagnoses",
    columns=[
        Column(
            name="PATIENT_HASH",
            order=1,
            data_type="text",
            length=12,
            primary_key=1,
            foreign_key="DEMOGRAPHICS.PATIENT_HASH",
        ),
        Column(name="DATE_OF_RECORDING", order=2, data_type="date", primary_key=2),
        Column(
            name="DIAGNOSIS_CODE",
            order=3,
            data_type="text",
            length=6,
            primary_key=3,
        ),
        Column(
            name="PRIMARY_DIAGNOSIS",
            order=4,
            data_type="text",
            length=1,
            enumerations={"Y": "Primary Diagnosis", "N": "Comorbidity"},
        ),
    ],
    description="ICD diagnoses for synthetic patients",
)

demo_dictionary = Dictionary(
    name="Demo Dictionary - Diagnoses",
    version="v1.0",
    version_notes="Dictionary created in demo",
    inclusion_criteria="* synthetic patients",
    exclusion_criteria="* real patients",
    tables=[diagnoses],
)
display(demo_dictionary)
```
```
Table(name='DIAGNOSES', description='ICD diagnoses for synthetic patients', 
 - Column(name='PATIENT_HASH', order=1, data_type='Text(12)', primary_key=1)
 - Column(name='DATE_OF_RECORDING', order=2, data_type='Date', primary_key=2)
 - Column(name='DIAGNOSIS_CODE', order=3, data_type='Text(6)', primary_key=3)
 - Column(name='PRIMARY_DIAGNOSIS', order=4, data_type='Text(1)'))
```
Once created, some useful helper functions are available and may be worth exploring:

 - `dictionary.add_table()`
 - `dictionary.get_table_names()`
 - `dictionary.remove_table()`
 - `dictionary.export_dictionary()`
 - `dictionary['table'].add_column()`
 - `dictionary['table'].remove_column()`
 - `dictionary['table'].get_primary_keys()`

### Generation From a Dataset
Dictionaries can also be generated automatically from a dataset, which can then be inspected and tweaked where necessary. Note that primary keys are not inferred, so should be fed in manually.


```python
from valediction import demo, Dataset

# Create Dataset
path_to_data = demo.DEMO_DATA
dataset = Dataset.create_from(path_to_data)

dataset.generate_dictionary("Demo Dictionary")
dataset.dictionary.set_primary_keys({
    "DEMOGRAPHICS": ["PATIENT_HASH"],
    "DIAGNOSES": ["PATIENT_HASH", "DATE_OF_RECORDING", "DIAGNOSIS_CODE"],
    "LAB_TESTS": ["PATIENT_HASH", "SAMPLE_DATE", "RESULT_DATE", "TEST_TYPE"],
    "VITALS": ["PATIENT_HASH", "OBSERVATION_TIME", "OBSERVATION_TYPE"],
})
dataset.dictionary.check() # check integrity
```
```
Generating dictionary for 4 tables
Generating DEMOGRAPHICS:  100%|██████████| 41/41 [00:00<00:00, 433.03step/s, Completed (0.0s)]                      
Generating DIAGNOSES:     100%|██████████| 33/33 [00:00<00:00, 441.31step/s, Completed (0.0s)]                              
Generating LAB_TESTS:     100%|██████████| 97/97 [00:00<00:00, 387.12step/s, Completed (0.2s)]                                
Generating VITALS:        100%|██████████| 33/33 [00:00<00:00, 413.70step/s, Completed (0.0s)]
```

It's worth exploring some arguments:
 - `feedback`: turn off progress for small performance improvement
 - `debug`: when True, provides full log of type inference with reasoning
 - `chunk_size`: where the `Dataset` contains items pointing to a `Path`, read the data in chunks to optimise RAM
 - `sample_rows`: only sample a specific number of rows (note that this overrides `chunk_size` and reads in a single chunk, and may result in a dictionary that is not fully reflective of the whole dataset)

A log can be explored by calling `dataset['table_name'].dictionary_runtimes`.

## 🦉 Data Validation <a id="data_validation"></a>
Once a `Dataset` is established and a `Dictionary` is attached, we can validate our dataset. Validation checks for:
* **Table & column structure** - ensuring the dataset schema matches the defined dictionary
* **Primary key integrity** - detecting nulls and collisions
* **Primary key whitespace** - flagging unintended whitespace within PKs
* **Data types** - identifying data mismatches and inconsistent date formats
* **Text lengths** - checking maximum permitted character lengths
* **Forbidden characters** - scanning for disallowed symbols

Here we establish a `Dataset`, import & attach a `Dictionary`, validate, and check:

```python
from valediction import Dataset, demo

PATH_TO_DATA = demo.DEMO_DATA
PATH_TO_DICT = demo.DEMO_DICTIONARY

# Create Dataset & Import Dictionary
dataset = Dataset.create_from(PATH_TO_DATA)
dataset.import_dictionary(PATH_TO_DICT)

# Optionally import data
dataset.import_data()

# Validate Dataset
dataset.validate()
dataset.check()
```

Validation arguments are as follows:
 - `feedback`: turn off progress for small performance improvement
 - `chunk_size`: where the `Dataset` contains items pointing to a `Path`, validate the data in chunks to optimise RAM

Validation runtimes can be explored by calling `dataset['table_name'].validation_runtimes`


## 🐞 Issue Inspection <a id="issue_inspection"></a>
As validation progresses, issues will be logged against each `DatasetItem` and additionally collected under the `Dataset`.

Let's import and validate the demo dataset, but tweak the dictionary to create some issues:

```python
from valediction import Dataset, demo, Config

PATH_TO_DATA = demo.DEMO_DATA
PATH_TO_DICT = demo.DEMO_DICTIONARY

# Create Dataset & Import Dictionary
dataset = Dataset.create_from(PATH_TO_DATA)
dataset.import_dictionary(PATH_TO_DICT)

# Change dictionary to create issues
dataset.dictionary['DEMOGRAPHICS']['SEX'].length = 5 # shorten max length for sex
dataset.dictionary['DIAGNOSES']['DIAGNOSIS_CODE'].primary_key = None # remove one of the columns from the PK

# Validate with issues
with Config() as config:
    config.forbidden_characters = ["y"] # make 'y' a forbidden character
    dataset.validate()
    dataset.check()
```
```
Validating 4 tables
Validating DEMOGRAPHICS:  100%|██████████| 15/15 [00:00<00:00, 286.68step/s, Completed with issues (0.0s)]   
WARNING: Issues detected in DEMOGRAPHICS.
Issues(
 - Issue(type='TextTooLong', table='DEMOGRAPHICS', column='SEX', total=45)
)
Validating DIAGNOSES:     100%|██████████| 15/15 [00:00<00:00, 236.71step/s, Completed with issues (0.0s)]   
WARNING: Issues detected in DIAGNOSES.
Issues(
 - Issue(type='PrimaryKeyCollision', table='DIAGNOSES', total=649)
)
Validating LAB_TESTS:     100%|██████████| 15/15 [00:00<00:00, 181.47step/s, Completed (0.0s)]               
Validating VITALS:        100%|██████████| 15/15 [00:00<00:00, 230.41step/s, Completed with issues (0.0s)]   
WARNING: Issues detected in VITALS.
Issues(
 - Issue(type='ForbiddenCharacter', table='VITALS', column='OBSERVATION_TYPE', total=334)
)

WARNING: Unvalidated tables or issues detected in DEMOGRAPHICS,DIAGNOSES,VITALS:
Issues(
 - Issue(type='TextTooLong', table='DEMOGRAPHICS', column='SEX', total=45)
 - Issue(type='PrimaryKeyCollision', table='DIAGNOSES', total=649)
 - Issue(type='ForbiddenCharacter', table='VITALS', column='OBSERVATION_TYPE', total=334)
)
```
```python                  
---------------------------------------------------------------------------
DataIntegrityError                        Traceback (most recent call last)
Cell In[2], line 18
     16 config.forbidden_characters = ["y"] # make 'y' a forbidden character
     17 dataset.validate()
---> 18 dataset.check()

File ~\Offline\Coding\Kitchen\valediction\src\valediction\datasets\datasets.py:789, in Dataset.check(self, readout)
    787         print_bold_red(f"\n{error}")
    788         print_red(self.issues)
--> 789     raise DataIntegrityError(f"{error}\n{self.issues}")
    790 else:
    791     return True

DataIntegrityError: WARNING: Unvalidated tables or issues detected in DEMOGRAPHICS,DIAGNOSES,VITALS:
Issues(
 - Issue(type='TextTooLong', table='DEMOGRAPHICS', column='SEX', total=45)
 - Issue(type='PrimaryKeyCollision', table='DIAGNOSES', total=649)
 - Issue(type='ForbiddenCharacter', table='VITALS', column='OBSERVATION_TYPE', total=334)
)
```

These issues can be inspected from the `Dataset` via indexing (`dataset.issues[2]`) or selecting (`dataset.issues.get("VITALS", "OBSERVATION_TYPE")[0]`), and each can be inspected to return the relevant values:

```python
# Inspect the forbidden character issue
forbidden_char_issue = dataset.issues.get("VITALS", "OBSERVATION_TYPE")[0]
forbidden_char_issue.inspect()
```
```
Issue(type='ForbiddenCharacter', table='VITALS', column='OBSERVATION_TYPE', total=334)

    OBSERVATION_TYPE
3	Systolic_BP
5	Systolic_BP
7	Systolic_BP
8	Systolic_BP
9	Respiratory_Rate
...	...
980	Systolic_BP
984	Respiratory_Rate
989	Respiratory_Rate
995	Respiratory_Rate
997	Respiratory_Rate
334 rows × 1 columns
```

We can include additional columns by passing in a list, or `True` to include all of them.
```python
# Inspect
dataset.issues[1].inspect(additional_columns=True)
```
```
Issue(type='PrimaryKeyCollision', table='DIAGNOSES', total=649):

	PATIENT_HASH	DATE_OF_RECORDING	DIAGNOSIS_CODE	PRIMARY_DIAGNOSIS
0	PCB1AC7AEFBC	02/03/2015	        F32	            Y
1	PCB1AC7AEFBC	02/03/2015	        J45	            N
2	PCB1AC7AEFBC	02/03/2015	        K35	            Y
3	PCB1AC7AEFBC	02/03/2015	        I10	            N
4	PCB1AC7AEFBC	02/03/2015	        C50	            Y
...	...	            ...	                ...	            ...
644	P00372442559	02/03/2015	        I21	            Y
645	P00372442559	02/03/2015	        M54.5	        Y
646	P00372442559	02/03/2015	        G40	            Y
647	P00372442559	02/03/2015	        E11	            N
648	P00372442559	02/03/2015	        J45	            N

```
We can see that tables with issues do not pass validation:
```python
display(dataset)
```
```
Dataset(len=4, dictionary_loaded=True, 
 - DatasetItem(name='DEMOGRAPHICS', data=Path('DEMOGRAPHICS.csv'), validated=False)
 - DatasetItem(name='DIAGNOSES', data=Path('DIAGNOSES.csv'), validated=False)
 - DatasetItem(name='LAB_TESTS', data=Path('LAB_TESTS.csv'), validated=True)
 - DatasetItem(name='VITALS', data=Path('VITALS.csv'), validated=False)
)
```
## 🤖 Convenience <a id="convenience"></a>
For convenience, the entire validation process can be run in a single line with a function that wraps around the class creation methods. A dataset (folder or `dict` of _name: `DataFrame`_ pairs) can be provided alongside the dictionary (`Dictionary` or dictionary filepath).

```python
import valediction as vale
from valediction import demo

files = demo.DEMO_DATA
dd = demo.DEMO_DICTIONARY

# Run without issues
dataset = vale.validate(files, dd)
dataset.check()

# Run with issues
with vale.Config() as config:
    config = vale.get_config()
    config.forbidden_characters=["b"]
    dataset = vale.validate(files, dd)

dataset.check()
```

```
Validating 4 tables
Validating DEMOGRAPHICS:  100%|██████████| 15/15 [00:00<00:00, 207.89step/s, Completed (0.0s)]                  
Validating DIAGNOSES:     100%|██████████| 15/15 [00:00<00:00,.37step/s, Completed (0.0s)]                      
Validating LAB_TESTS:     100%|██████████| 15/15 [00:00<00:00, 199.13step/s, Completed (0.0s)]                        
Validating VITALS:        100%|██████████| 15/15 [00:00<00:00, 191.11step/s, Completed (0.0s)]                      
```

```
Validating 4 tables
Validating DEMOGRAPHICS:  100%|██████████| 15/15 [00:00<00:00, 260.10step/s, Completed (0.0s)]               
Validating DIAGNOSES:     100%|██████████| 15/15 [00:00<00:00, 236.36step/s, Completed (0.0s)]                   
Validating LAB_TESTS:     100%|██████████| 15/15 [00:00<00:00, 184.56step/s, Completed with issues (0.0s)]         
WARNING: Issues detected in LAB_TESTS.
Issues(
 - Issue(type='ForbiddenCharacter', table='LAB_TESTS', column='TEST_TYPE', total=144)
)
Validating VITALS:        100%|██████████| 15/15 [00:00<00:00, 237.49step/s, Completed (0.0s)]                   

WARNING: Unvalidated tables or issues detected in LAB_TESTS:
Issues(
 - Issue(type='ForbiddenCharacter', table='LAB_TESTS', column='TEST_TYPE', total=144)
)
```

```python
---------------------------------------------------------------------------
DataIntegrityError                        Traceback (most recent call last)
Cell In[3], line 17
     14     config.forbidden_characters=["b"]
     15     dataset = vale.validate(files, dd)
---> 17 dataset.check()

File ~\Offline\Coding\Kitchen\valediction\src\valediction\datasets\datasets.py:789, in Dataset.check(self, readout)
    787         print_bold_red(f"\n{error}")
    788         print_red(self.issues)
--> 789     raise DataIntegrityError(f"{error}\n{self.issues}")
    790 else:
    791     return True

DataIntegrityError: WARNING: Unvalidated tables or issues detected in LAB_TESTS:
Issues(
 - Issue(type='ForbiddenCharacter', table='LAB_TESTS', column='TEST_TYPE', total=144)
)
```

## ⚡ Full Run <a id="full_run"></a>
Here's a full run:
```python
from valediction import Dataset, demo

PATH_TO_DATA = demo.DEMO_DATA
PATH_TO_DICT = demo.DEMO_DICTIONARY

# Create Dataset & Import Dictionary
dataset = Dataset.create_from(PATH_TO_DATA)
dataset.import_dictionary(PATH_TO_DICT)

# Validate
dataset.validate()

# Import Data
dataset.import_data()
display(dataset)
```

### Kraken Integration
If using [Kraken](https://github.com/SETT-Centre-Data-and-AI/Kraken) to power your reproducible analytical pipeline, you can easily integrate the two packages, with a full run as follows:
```python
import kraken
from valediction import Dataset

# Point to Data Dictionary
path_dictionary = r'C:/...path_to_dictionary'

# Extract data from databases
sql_folder = r'C:/...path_to_sql'
results = kraken.run(sql_folder)

# Create Dataset & Import Dictionary
dataset = Dataset.create_from(results.convert_to_dict())
dataset.import_dictionary(path_dictionary)

# Validate
dataset.validate(chunk_size=1_000_000)

# Import Data
dataset.import_data()
display(dataset)
```
Alternatively:
```python
import kraken
from valediction import validate 

# Point to Data Dictionary
path_dictionary = r'C:/...path_to_dictionary'

# Extract data from databases
sql_folder = r'C:/...path_to_sql'
results = kraken.run(sql_folder)

# Convenience Validation
dataset = validate(dataset=results.convert_to_dict(), dictionary=path_dictionary)
dataset.import_dictionary(path_dictionary)
dataset.check()
```
