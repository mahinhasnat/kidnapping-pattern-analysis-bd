# Kidnapping Pattern Analysis — Bangladesh 🇧🇩

An AI/ML-based research case study analyzing patterns in child kidnapping
and missing-child cases in Bangladesh — identifying which age groups,
locations, and circumstances carry the highest risk, in order to support
awareness and prevention efforts.

> ⚠️ **Disclaimer:** The national-level statistics in this project are
> real and drawn from verified news/police sources, with every source
> cited. However, `data/case_study_dataset.csv` is a **simulated
> dataset** — it contains no real individual victim records. This is a
> research/educational proof-of-concept, not an operational surveillance
> tool.

---

## 📌 Purpose

Child safety has become a serious national concern in Bangladesh.
According to Police Headquarters data, 15,917 children were reported
missing between January and July 2026 — an average of more than 75 per
day. This project uses data analysis and machine learning to understand
the patterns behind this crisis, with the goal of supporting evidence-based
awareness and prevention efforts. It is not intended to assign blame, but
to build data-driven awareness.

## 🗂️ Project Structure

```
kidnapping-pattern-analysis-bd/
├── README.md
├── data/
│   ├── real_national_statistics.csv     # Real, cited national statistics
│   ├── case_study_dataset.csv           # Simulated case-level dataset (input)
│   └── case_study_dataset_analyzed.csv  # Output with cluster/risk labels
├── src/
│   ├── generate_dataset.py       # Script that generates the simulated dataset
│   └── analysis.py               # EDA + clustering + classifier pipeline
└── outputs/
    └── eda_charts.png            # Visualization charts
```

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/mahinhasnat/kidnapping-pattern-analysis-bd.git
cd kidnapping-pattern-analysis-bd

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib

# 3. Generate the dataset
python3 src/generate_dataset.py

# 4. Run the analysis
python3 src/analysis.py
```

Running `analysis.py` prints the full EDA, clustering, and classifier
report to the terminal, and generates `outputs/eda_charts.png` and
`case_study_dataset_analyzed.csv`.

---

# 📄 Full Case Study Report

## 1. Introduction

Child safety has become a pressing national concern in Bangladesh. This
case study demonstrates how an AI/ML pipeline can be used to analyze
reported kidnapping/missing-child cases to identify **which age groups,
locations, and circumstances carry the highest risk**, supporting
evidence-based prevention and awareness efforts.

## 2. Real-World Context (verified, cited sources)

| Metric | Value | Source |
|---|---|---|
| Children reported missing, Jan–Jul 2026 | 15,917 | Police Headquarters / ISHR |
| Average missing children per day | 75+ | Police Headquarters / ISHR |
| Children killed, Jan–Aug 2026 | 155 | Human rights report (Dhaka Tribune) |
| Children still untraced (of ~15,000 missing) | 2,038 (~13%) | Human rights report |
| Abduction cases, 2021 → 2025 | 445 → 1,005 (more than doubled) | Police statistics |
| Highest-abduction division | Dhaka (403 cases), then Chattogram (203) | Police statistics |
| Sharpest monthly spike | June 2026: 3,163 missing-children reports | ISHR |

**Key reported pattern:** across multiple documented cases (e.g. the 2026
Pallabi case, the 2025 Magura case), the accused was **known to the
victim** — a neighbor or relative — rather than a stranger. Law enforcement
has also noted that many abduction cases stem from ransom or business/
financial disputes.

> Note: No open, case-level government dataset is currently public. The
> figures above are real and cited; anything below the case level
> (individual records) is simulated for pipeline demonstration — see
> Section 3.

## 3. Methodology

Since granular, case-level open data isn't publicly available, this case
study uses a **simulated dataset (500 cases)** whose statistical shape
(division split, monthly seasonality, relation-to-accused distribution) is
deliberately calibrated to match the real, cited patterns above. This lets
the full pipeline be built and tested now, and swapped for real
(anonymized) records later — e.g. from Ain o Salish Kendra (ASK), RTI
requests to police, or partner NGOs.

**Pipeline steps:**
1. Data cleaning & age-group bucketing (0–5, 6–9, 10–13, 14–17)
2. Descriptive/EDA analysis (age, division, month, relation-to-accused)
3. **K-Means clustering** — groups cases into "profiles" sharing similar
   characteristics
4. **Random Forest classifier** — predicts high-risk outcome (still
   missing/deceased vs. rescued) from case features, and ranks which
   features matter most

## 4. Findings

- **Highest-risk age group: 6–9 years** (41.8% of simulated cases), closely
  followed by 10–13 (36.8%)
- **78.4%** of cases involved someone known to the victim (neighbor,
  relative, acquaintance, or domestic employer) rather than a stranger —
  consistent with real reported cases
- **Dhaka division** accounted for the largest share of cases, matching the
  real police statistics (403 of ~1,005 cases nationally)
- **June** showed a sharp seasonal spike — matching the real ISHR-reported
  spike of 3,163 cases that month
- The Random Forest model ranked **age** and **month** as the strongest
  predictors of a high-risk outcome, followed by division and relation to
  accused

See `outputs/eda_charts.png` for visualizations and
`data/case_study_dataset_analyzed.csv` for the full cluster/risk-labeled
data.

## 5. Recommendations

1. **Targeted awareness campaigns** for parents/guardians of 6–13
   year-olds, the highest-risk age band
2. **"Known-perpetrator" safety education** — since most cases involve
   someone known to the child, awareness should extend beyond
   stranger-danger messaging
3. **Seasonal resource allocation** — pre-position police/NGO response
   capacity ahead of high-risk months (e.g. June, coinciding with school
   holidays)
4. **Division-level focus** on Dhaka and Chattogram, without neglecting
   smaller divisions
5. **Push for open, anonymized case-level data** from police/ASK so real
   models (not simulated ones) can be trained and validated

## 6. Limitations

- Case-level data is **simulated**, shaped to match real aggregate
  statistics but not actual records — findings are illustrative of the
  *method*, not a real-world risk certification
- Real deployment would need ethics/privacy review before using any actual
  victim data (anonymization, consent, data protection law compliance)
- Model performance (63% accuracy, weak recall on "high risk" class) shows
  more/better features and real data are needed before any operational use

## 7. Data Sources

- Police Headquarters, Bangladesh (via ISHR report)
- International Society for Human Rights (ISHR)
- Dhaka Tribune / human rights organization report (Jan–Aug 2026)
- Bonikbarta — police abduction statistics (2021–2025)

## 8. Files in This Repository

| File | Purpose |
|---|---|
| `data/real_national_statistics.csv` | Real, cited national/division statistics |
| `src/generate_dataset.py` | Generates the simulated case-level dataset |
| `data/case_study_dataset.csv` | The simulated case-level dataset (input) |
| `src/analysis.py` | Full EDA + clustering + classifier pipeline |
| `data/case_study_dataset_analyzed.csv` | Output data with cluster/risk labels |
| `outputs/eda_charts.png` | Visualizations (age, division, month, relation) |

---

## 📝 License

Free to use for educational and research purposes. Any reuse with real
data must comply with applicable privacy laws.

## 🤝 Contributing

Contributions of real, anonymized case-level data or additional analysis
are welcome via Pull Request.
