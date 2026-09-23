from income_pipeline.generate import build_records, positive_probability


def test_label_rule_increases_with_education_age_and_hours():
    low = positive_probability(18, "Preschool", 10, "Gov")
    high = positive_probability(45, "Doctorate", 60, "Private")
    assert high > low
    assert high <= 0.95


def test_generator_is_deterministic_for_a_seed():
    assert build_records(30, seed=1) == build_records(30, seed=1)
    assert build_records(30, seed=1) != build_records(30, seed=2)
