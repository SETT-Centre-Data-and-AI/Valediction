<p align="center">
  <img src="docs/images/valediction.png" alt="Valediction", width=300>
</p>

# Valediction
_**Data dictionary–driven validation for reproducible analytical pipelines.**_

**Valediction** is a lightweight data validation package that supports **generation, import, and constraint enforcement** of user-defined **data dictionaries** against datasets.

Developed by the **Data & AI Research (DAIR) Unit** at University Hospital Southampton NHSFT for use in clinical research, Valediction integrates into **reproducible analytical pipelines** to ensure data integrity throughout **data transfer and transformation**.

Data dictionaries are interoperable between Python and MS Excel, supporting both collaboration and pipeline version control. 

**Features**:
* **Translates** data dictionaries between MS Excel and Python
* **Generates** data dictionaries automatically from datasets 
* **Enforces** dictionary constraints against datasets
* **Validates** large datasets in full, or via iterated chunk streaming for efficient RAM management
* **Logs and reports deviations** from the dictionary, enabling user inspection and correction

**Valediction enforces**:
* **Table & column structure** - ensuring the dataset schema matches the defined dictionary
* **Primary key integrity** - detecting nulls and collisions
* **Primary key whitespace** - flagging unintended whitespace within PKs
* **Data types** - identifying data mismatches and inconsistent date formats
* **Text lengths** - checking maximum permitted character lengths
* **Forbidden characters** - scanning for disallowed symbols

# 🧭 Resources
* [Installation](./docs/installation-and-setup-guide.md)
* [Usage Guide](./docs/usage-guide.md)
* [Kraken](https://github.com/SETT-Centre-Data-and-AI/Kraken) (_for end-to-end automated pipeline integration_)
* [Data Dictionary Template](./src/valediction/dictionary/template/PROJECT%20-%20Data%20Dictionary.xltx) (_download_)

# ⚡ Quickstart & Demo
1) Install: `pip install valediction`
2) Download the [Data Dictionary Template](./src/valediction/dictionary/template/PROJECT%20-%20Data%20Dictionary.xltx)
3) Complete and save an `.xlsx` data dictionary for your dataset
4) Import dataset & data dictionary, and validate:

```python
from valediction import Dataset

# Demo (replace appropriately)
from valediction import demo
PATH_TO_DATA = demo.DIR_DEMO # folder containing data .csv files
PATH_TO_DICT = demo.PATH_DICTIONARY_DEMO # accompanying data dictionary
```
```python
# Create Dataset & Import Dictionary
dataset = Dataset.create_from(PATH_TO_DATA)
dataset.import_dictionary(PATH_TO_DICT)
dataset
```
```python
# Validate Dataset
dataset.validate(chunk_size=None)
dataset.issues
```

# 🧠 Function Quicklist
### Dictionaries
* `import_dictionary()` - load an `.xlsx` data dictionary into a Python Dictionary object
* `export_dictionary()` - export a Dictionary to `.xlsx` 
* `Dictionary() / Table() / Column()` - Python driven dictionary construction

### Dataset Creation
* `Dataset.create_from()` - establish a `Dataset`, either from a folder of `.csv` files, or by directly feeding tuples of `DataFrame`s and table names as a list

### Dataset Control
_Dictionary functions are callable directly from a `Dataset` object:_
* `dataset.import_data()` - optionally load `DataFrame`s from the data
* `dataset.import_dictionary()` - attach a prepared data dictionary from an `.xlsx`
* `dataset.generate_dictionary()` - automatically generate a dictionary from a given dataset, including data types and 
* `dataset.export_dictionary()` - export the `Dataset`'s dictionary
* `dataset.validate()` - validate data against the attached dictionary

### Reporting
* `dataset.issues` - inspect issues
* `issue.inspect()` - highlight problematic data values
* `dataset_item.dictionary_runtimes` - runtime log for dictionary generation
* `dataset_item.validation_runtimes` - runtime log for validation


# 🤝 Contributing
Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

# ⚖️ License
This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].
[![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
[SETT]: https://github.com/SETT-Centre-Data-and-AI

# 🧑‍🔬 Authors
Valediction was developed by Cai Davis at University Hospital Southampton NHSFT's Data & AI Research Unit (DAIR) - part of the [Southampton Emerging Therapies and Technology (SETT) Centre][SETT].
<p align="center">
  <a href="https://github.com/SETT-Centre-Data-and-AI">
    <img src="docs/images/SETT Header.png" alt="NHS UHS SETT Centre">
  </a>
</p>
