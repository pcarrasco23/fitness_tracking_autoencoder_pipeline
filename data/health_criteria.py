# Single source of truth for age-adjusted healthy metric ranges.
# Both health scoring and synthetic data generation use these functions
# so that the criteria are always consistent.

DEFAULT_AGE = 40  # Used when age is not available


def steps_range(age):
    # WHO/CDC daily step recommendations by age:
    #   < 18: 8,000–16,000  (children and teens are more active)
    #   18–40: 7,000–15,000  (active adult range)
    #   40–60: 6,000–14,000  (moderate reduction with age)
    #   60–75: 5,000–12,000  (older adults, lower baseline)
    #   75+:   3,500–10,000  (senior range)
    age = age if age is not None else DEFAULT_AGE
    if age < 18:
        return 8000, 16000
    if age < 40:
        return 7000, 15000
    if age < 60:
        return 6000, 14000
    if age < 75:
        return 5000, 12000
    return 3500, 10000


def heart_rate_range(age):
    # Normal resting heart rate (bpm) per AHA guidelines:
    #   < 60: 60–100 bpm  (standard adult range)
    #   60+:  60–105 bpm  (slightly wider for older adults)
    age = age if age is not None else DEFAULT_AGE
    if age < 60:
        return 60, 100
    return 60, 105


def sleep_range(age):
    # Recommended nightly sleep hours per NSF/CDC guidelines:
    #   < 18: 8–10 hrs  (teenagers need more sleep)
    #   18–65: 7–9 hrs  (standard adult recommendation)
    #   65+:   7–8 hrs  (older adults typically need slightly less)
    age = age if age is not None else DEFAULT_AGE
    if age < 18:
        return 8, 10
    if age < 65:
        return 7, 9
    return 7, 8


def exercise_range(age):
    # Daily moderate exercise (minutes) based on WHO guidelines:
    #   < 40: 20–60 min  (younger adults, higher capacity)
    #   40–60: 18–50 min  (moderate reduction)
    #   60–75: 15–45 min  (older adults, lower intensity acceptable)
    #   75+:   10–30 min  (seniors, focus on movement over intensity)
    age = age if age is not None else DEFAULT_AGE
    if age < 40:
        return 20, 60
    if age < 60:
        return 18, 50
    if age < 75:
        return 15, 45
    return 10, 30


def stress_range(age):
    # Healthy stress level score (scale 1–10):
    #   < 60: 1–5  (moderate stress tolerance)
    #   60+:  1–4  (lower threshold; chronic stress is higher risk in older adults)
    age = age if age is not None else DEFAULT_AGE
    if age < 60:
        return 1, 5
    return 1, 4


def bmi_range(age):
    # Healthy BMI (kg/m²) per WHO/CDC guidelines:
    #   < 40: 18.5–24.9  (standard healthy range)
    #   40–60: 18.5–26.0  (slightly wider; modest weight gain is acceptable)
    #   60+:  20.0–27.0  (higher lower bound; low BMI increases frailty risk)
    age = age if age is not None else DEFAULT_AGE
    if age < 40:
        return 18.5, 24.9
    if age < 60:
        return 18.5, 26.0
    return 20.0, 27.0


def weight_kg_range(age):
    # Derived from bmi_range applied to typical average heights by age group:
    #   < 18: ~1.60 m avg  →  BMI 18.5–24.9  →  47–64 kg
    #   18–40: ~1.70 m avg →  BMI 18.5–24.9  →  53–72 kg
    #   40–60: ~1.68 m avg →  BMI 18.5–26.0  →  52–73 kg
    #   60+:   ~1.65 m avg →  BMI 20.0–27.0  →  54–73 kg
    age = age if age is not None else DEFAULT_AGE
    if age < 18:
        return 47.0, 64.0
    if age < 40:
        return 53.0, 72.0
    if age < 60:
        return 52.0, 73.0
    return 54.0, 73.0


def calories_burned_range(age):
    # Healthy daily calories burned (kcal) based on typical TDEE for moderately active
    # individuals at a healthy BMI, per Mifflin-St Jeor estimates:
    #   < 18: 1,800–2,800 kcal  (teens have higher metabolic needs)
    #   18–40: 1,800–3,000 kcal  (peak adult metabolic rate)
    #   40–60: 1,600–2,600 kcal  (metabolism slows ~5% per decade after 40)
    #   60+:   1,400–2,200 kcal  (further reduction with age and muscle loss)
    age = age if age is not None else DEFAULT_AGE
    if age < 18:
        return 1800, 2800
    if age < 40:
        return 1800, 3000
    if age < 60:
        return 1600, 2600
    return 1400, 2200
