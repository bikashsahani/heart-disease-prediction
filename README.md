
## heart-disease-analysis
End-to-end Exploratory Data Analysis on a Heart Disease dataset
(1,000 patients, 16 clinical & lifestyle features) to identify
key risk factors using Python. Cleaned data was also loaded into
a PostgreSQL database.

## Dataset
- Source: Kaggle
- Shape: 1,000 rows × 16 columns
- Target: Heart Disease (No = 608, Yes = 392)
- Missing values: 340 in Alcohol Intake column (handled via fillna)

## Features Analysed
Age, Gender, Cholesterol, Blood Pressure, Heart Rate, Smoking,
Alcohol Intake, Exercise Hours, Family History, Diabetes, Obesity,
Stress Level, Blood Sugar, Exercise Induced Angina, Chest Pain Type

## Key Findings
- Age is the strongest predictor of heart disease (correlation: 0.65)
- Cholesterol shows moderate correlation (0.37)
- Asymptomatic chest pain is most linked to heart disease
- Patients with family history have significantly higher risk
- Less active patients (lower exercise hours) trend toward heart disease
- Stress Level and Blood Pressure showed no strong correlation

## Tools Used
Python | Pandas | NumPy | Matplotlib | Seaborn | PostgreSQL | psycopg2 | SQLAlchemy

## How to Run
pip install pandas, numpy, matplotlib.pyplot, seaborn, psycopg2, sqlalchemy, jupyter
jupyter notebook heart_disease_analysis.ipynb

## Files
heart_disease_analysis.ipynb — Main EDA notebook
heart_disease_dataset.csv   — Raw dataset
Cleaned_Heart_Disease.csv   — Cleaned output

## 👤 Author
Bikash Sahani
github.com/bikashsahani | linkedin.com/in/bikash-sahani