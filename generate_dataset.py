"""
Generates an ILLUSTRATIVE / SIMULATED case-level dataset for the case study.
This is NOT real individual case data (no such open dataset exists publicly).
It is shaped using the real, cited aggregate patterns:
  - Division split roughly: Dhaka ~40%, Chattogram ~20%, Other ~40%
  - Monthly seasonality with a sharp spike in June
  - Perpetrator-known-to-victim pattern emphasized in reporting
Use this purely to demonstrate/test the analysis pipeline. Replace with
real (anonymized) records if/when you obtain them from a proper source
(e.g. Ain o Salish Kendra, police data via RTI, NGO datasets).
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 500

divisions = ["Dhaka", "Chattogram", "Rajshahi", "Khulna", "Barishal", "Sylhet", "Rangpur", "Mymensingh"]
division_weights = [0.40, 0.20, 0.09, 0.08, 0.06, 0.06, 0.06, 0.05]

months = list(range(1, 13))
month_weights = np.array([0.06,0.06,0.06,0.07,0.08,0.28,0.09,0.07,0.06,0.06,0.06,0.05])
month_weights = month_weights / month_weights.sum()

relations = ["Neighbor", "Relative", "Family friend/acquaintance", "Stranger", "Domestic worker/employer"]
relation_weights = [0.30, 0.22, 0.18, 0.20, 0.10]

genders = ["Female", "Male"]
gender_weights = [0.62, 0.38]

outcomes = ["Rescued/Recovered", "Still missing", "Deceased"]
outcome_weights = [0.68, 0.22, 0.10]

modes = ["Lured/deceived", "Taken from home", "Taken from public place", "Ransom abduction", "Trafficking suspected"]
mode_weights = [0.32, 0.20, 0.22, 0.16, 0.10]

age = rng.normal(loc=9, scale=3.2, size=N).clip(1, 17).round().astype(int)
division = rng.choice(divisions, size=N, p=division_weights)
month = rng.choice(months, size=N, p=month_weights)
relation = rng.choice(relations, size=N, p=relation_weights)
gender = rng.choice(genders, size=N, p=gender_weights)
outcome = rng.choice(outcomes, size=N, p=outcome_weights)
mode = rng.choice(modes, size=N, p=mode_weights)
year = rng.choice([2024, 2025, 2026], size=N, p=[0.25, 0.35, 0.40])

df = pd.DataFrame({
    "case_id": [f"C{i+1:04d}" for i in range(N)],
    "year": year,
    "month": month,
    "age": age,
    "gender": gender,
    "division": division,
    "relation_to_accused": relation,
    "mode_of_abduction": mode,
    "outcome": outcome,
})

df.to_csv("case_study_dataset.csv", index=False)
print(df.head(10).to_string(index=False))
print("\nSaved case_study_dataset.csv with", len(df), "rows")
