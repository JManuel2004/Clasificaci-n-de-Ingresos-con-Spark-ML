"""Shared contracts for the income classification pipeline."""

POSITIVE_LABEL = ">50K"
NEGATIVE_LABEL = "<=50K"
LABELS = (NEGATIVE_LABEL, POSITIVE_LABEL)

# Ordered so the rank is a meaningful education scale, not a frequency code.
EDUCATION_ORDINAL = {
    "Preschool": 1,
    "9th": 2,
    "10th": 3,
    "11th": 4,
    "12th": 5,
    "HS-grad": 6,
    "Some-college": 7,
    "Assoc-voc": 8,
    "Bachelors": 9,
    "Masters": 10,
    "Doctorate": 11,
}

SEX_VALUES = ("Female", "Male")
WORKCLASS_VALUES = ("Gov", "Private", "Self-emp")

MIN_AGE = 16
MAX_AGE = 100
MIN_HOURS = 1
MAX_HOURS = 99

# fnlwgt is the Adult census sampling weight. It adjusts how much a row
# represents the population; it is not a property of the person.
EXCLUDED_FEATURES = ("fnlwgt",)
EXCLUSION_REASON = (
    "fnlwgt is a census sampling weight, not an attribute of the person, "
    "so it is excluded from the feature vector."
)

CATEGORICAL_COLUMNS = ("sex", "workclass")
NUMERIC_MODEL_COLUMNS = (
    "age",
    "hours_per_week",
    "education_ordinal",
    "works_overtime",
)

RAW_COLUMNS = (
    "age",
    "sex",
    "workclass",
    "fnlwgt",
    "education",
    "hours_per_week",
    "label",
)
