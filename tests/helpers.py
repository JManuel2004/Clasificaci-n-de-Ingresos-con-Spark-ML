def person(**overrides) -> dict:
    row = {
        "age": 40,
        "sex": "Female",
        "workclass": "Private",
        "fnlwgt": 100_000,
        "education": "Bachelors",
        "hours_per_week": 40,
        "label": "<=50K",
    }
    row.update(overrides)
    return row
