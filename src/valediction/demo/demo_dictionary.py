from valediction.dictionary.model import Column, Dictionary, Table


def demo_dictionary() -> Dictionary:
    """Get a Dictionary object equivalent to the demo dataset.

    Returns:
        Dictionary: Demo Dictionary
    """
    demographics = Table(
        name="demographics",
        columns=[
            Column(
                name="PATIENT_HASH", order=1, data_type="text", length=12, primary_key=1
            ),
            Column(name="DATE_OF_BIRTH", order=2, data_type="date"),
            Column(name="ETHNICITY", order=3, data_type="text", length=1),
            Column(name="SEX", order=4, data_type="text", length=6),
            Column(name="DATE_OF_DEATH", order=5, data_type="date"),
        ],
        description="Demographic information for synthetic patients",
    )

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

    lab_tests = Table(
        name="lab_tests",
        columns=[
            Column(
                name="PATIENT_HASH",
                order=1,
                data_type="text",
                length=12,
                primary_key=1,
                foreign_key="DEMOGRAPHICS.PATIENT_HASH",
            ),
            Column(name="SAMPLE_DATE", order=2, data_type="date", primary_key=2),
            Column(name="RESULT_DATE", order=3, data_type="date", primary_key=3),
            Column(name="SAMPLE_TYPE", order=4, data_type="text", length=32),
            Column(
                name="TEST_TYPE", order=5, data_type="text", length=32, primary_key=4
            ),
            Column(name="RESULT_RAW", order=6, data_type="text", length=256),
            Column(name="UNITS", order=7, data_type="text", length=16),
            Column(name="RESULT_NUMERIC", order=8, data_type="float"),
            Column(
                name="RESULT_PROCESSED",
                order=9,
                data_type="text",
                length=8,
                enumerations={"positive": "Positive", "negative": "Negative"},
            ),
            Column(
                name="OPERATOR",
                order=10,
                data_type="text",
                length=1,
                enumerations={"-": "Less Than", "+": "Greater Than"},
            ),
            Column(name="RANGE_LOW", order=11, data_type="float"),
            Column(name="RANGE_HIGH", order=12, data_type="float"),
        ],
        description="Lab test results for synthetic patients",
    )

    vitals = Table(
        name="vitals",
        columns=[
            Column(
                name="PATIENT_HASH",
                order=1,
                data_type="text",
                length=12,
                primary_key=1,
                foreign_key="DEMOGRAPHICS.PATIENT_HASH",
            ),
            Column(
                name="OBSERVATION_TIME", order=2, data_type="datetime", primary_key=2
            ),
            Column(
                name="OBSERVATION_TYPE",
                order=3,
                data_type="text",
                length=32,
                primary_key=3,
            ),
            Column(name="RESULT", order=4, data_type="float"),
        ],
        description="Numeric vital sign obs for synthetic patients",
    )

    demo_dictionary = Dictionary(
        name="Synthetic Data",
        version="v0.1",
        version_notes="Synthethic dataset for allowing Wessex SDE onboarding",
        inclusion_criteria="* synthetic patients",
        exclusion_criteria="* real patients",
        tables=[demographics, diagnoses, lab_tests, vitals],
    )

    return demo_dictionary
